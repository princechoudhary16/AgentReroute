"""Conservative preflight dispatch. Original endpoint contracts remain intact."""
import copy
import inspect
import threading
import uuid
from collections import deque
from functools import wraps
from .catalog import PlanError, at, compile_plan, encoded, equal, schema_at, validate
from .adapters import is_async


class Blocked(RuntimeError):
    def __init__(self, code, tool, rule=None, message='Preflight validation did not establish the prerequisite'):
        self.code, self.tool, self.rule = code, tool, rule
        super().__init__(message)

    def to_dict(self):
        return {'status': 'blocked', 'code': self.code, 'tool': self.tool, 'rule': self.rule, 'message': str(self)}


class Middleware:
    def __init__(self, tools, plan, *, validators, mode='enforce', max_validations=16):
        if mode not in ('enforce', 'shadow'):
            raise PlanError('Mode must be enforce or shadow')
        if type(max_validations) is not int or not 1 <= max_validations <= 1024:
            raise PlanError('Validation budget must be 1..1024')
        validators = tuple(validators)
        self._tools, self._rules = compile_plan(tools, plan, validators)
        self.mode, self.max_validations = mode, max_validations
        self._events, self._lock = deque(maxlen=1000), threading.Lock()
        self._async_tools = {}
        def needs_async(name):
            if name not in self._async_tools:
                self._async_tools[name] = (is_async(self._tools[name].handler)
                    or any(needs_async(rule['validator']) for rule in self._rules[name]))
            return self._async_tools[name]
        for name in self._tools:
            needs_async(name)
        self.coverage = {'guarded_tools': [n for n, rules in self._rules.items() if rules],
                         'unguarded_tools': [n for n, rules in self._rules.items() if not rules],
                         'validators': list(validators), 'abstentions': copy.deepcopy(plan['abstentions'])}

    def definitions(self):
        return [tool.definition() for tool in self._tools.values()]

    def wrapped_tools(self):
        result = {}
        for name, tool in self._tools.items():
            if (self.mode == 'enforce' and self._async_tools[name]) or is_async(tool.handler):
                def bind_async(tool_name, handler):
                    @wraps(handler)
                    async def invoke(*args, **kwargs):
                        arguments = self._bound_call(tool_name, handler, args, kwargs)
                        return await self._async(tool_name, arguments, self._context(), lambda: handler(*args, **kwargs))
                    return invoke
                result[name] = bind_async(name, tool.handler)
            else:
                def bind(tool_name, handler):
                    @wraps(handler)
                    def invoke(*args, **kwargs):
                        arguments = self._bound_call(tool_name, handler, args, kwargs)
                        return self._sync(tool_name, arguments, self._context(), lambda: handler(*args, **kwargs))
                    return invoke
                result[name] = bind(name, tool.handler)
        return result

    def _bound_call(self, name, handler, args, kwargs):
        if self.mode == 'shadow' or (not args and not self._rules[name]):
            return kwargs
        try:
            signature = inspect.signature(handler)
        except (TypeError, ValueError):
            if args and self._rules[name]:
                raise Blocked('unsupported_binding_signature', name)
            return kwargs
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        for parameter in signature.parameters.values():
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                values.update(values.pop(parameter.name, {}))
        return values

    def audit(self):
        with self._lock:
            return copy.deepcopy(list(self._events))

    def _event(self, context, tool, decision, rule=None, code=None):
        event = {'dispatch_id': context['id'], 'tool': tool, 'decision': decision}
        if rule:
            event['rule'] = rule
        if code:
            event['code'] = code
        with self._lock:
            self._events.append(event)

    @staticmethod
    def _context():
        return {'id': uuid.uuid4().hex, 'cache': {}, 'count': 0}

    def _arguments(self, tool, arguments, rule):
        try:
            result = {name: copy.deepcopy(at(arguments, binding['from']) if 'from' in binding else binding['literal'])
                      for name, binding in rule['bindings'].items()}
            for name, binding in rule['bindings'].items():
                if 'from' in binding:
                    validate(result[name], schema_at(self._tools[tool].input_schema, binding['from']))
            encoded(result)
            validate(result, self._tools[rule['validator']].input_schema)
            return result
        except (KeyError, IndexError, TypeError, ValueError):
            raise Blocked('invalid_binding_input', tool, rule['id']) from None

    @staticmethod
    def _validator_unchanged(tool, rule, bound, key):
        try:
            if encoded(bound) == key[1]:
                return
        except (TypeError, ValueError):
            pass
        raise Blocked('validator_input_mutated', tool, rule['id'])

    def _target_unchanged(self, tool, arguments, checked):
        for rule, key in checked:
            try:
                if encoded(self._arguments(tool, arguments, rule)) == key[1]:
                    continue
            except Blocked:
                pass
            raise Blocked('arguments_changed', tool, rule['id'])

    def _evidence(self, tool, rule, result):
        try:
            encoded(result)
            validate(result, self._tools[rule['validator']].output_schema)
            if not all(equal(at(result, check['pointer']), check['equals']) for check in rule['success']):
                raise ValueError('Prerequisite failed')
        except (KeyError, IndexError, TypeError, ValueError):
            raise Blocked('validation_failed', tool, rule['id']) from None

    def _claim(self, context, tool, rule, arguments):
        key = (rule['validator'], encoded(arguments))
        if key not in context['cache']:
            if context['count'] >= self.max_validations:
                raise Blocked('validation_budget', tool, rule['id'])
            context['count'] += 1
        return key

    def dispatch(self, name, arguments):
        return self._sync(name, arguments, self._context())

    def _sync(self, name, arguments, context, invoke=None):
        tool = self._tools[name]
        if (self.mode == 'enforce' and self._async_tools[name]) or is_async(tool.handler):
            raise Blocked('async_required', name, message='Use adispatch for asynchronous tools')
        try:
            checked = []
            if self.mode == 'enforce':
                for rule in self._rules[name]:
                    bound = self._arguments(name, arguments, rule)
                    key = self._claim(context, name, rule, bound)
                    if key not in context['cache']:
                        try:
                            result = self._sync(rule['validator'], bound, context)
                            self._validator_unchanged(name, rule, bound, key)
                            context['cache'][key] = copy.deepcopy(result)
                        except Blocked:
                            raise
                        except Exception as exc:
                            raise Blocked('validator_error', name, rule['id'], f"Validator failed ({type(exc).__name__})") from None
                    self._evidence(name, rule, context['cache'][key])
                    checked.append((rule, key))
            self._target_unchanged(name, arguments, checked)
            result = invoke() if invoke is not None else tool.handler(**arguments)
            if inspect.isawaitable(result):
                if inspect.iscoroutine(result):
                    result.close()
                raise Blocked('async_required', name, message='Use adispatch for asynchronous tool results')
            self._event(context, name, 'shadow' if self.mode == 'shadow' else 'forwarded')
            return result
        except Blocked as exc:
            self._event(context, name, 'blocked', exc.rule, exc.code)
            raise

    async def adispatch(self, name, arguments):
        return await self._async(name, arguments, self._context())

    async def _async(self, name, arguments, context, invoke=None):
        tool = self._tools[name]
        try:
            checked = []
            if self.mode == 'enforce':
                for rule in self._rules[name]:
                    bound = self._arguments(name, arguments, rule)
                    key = self._claim(context, name, rule, bound)
                    if key not in context['cache']:
                        try:
                            result = await self._async(rule['validator'], bound, context)
                            self._validator_unchanged(name, rule, bound, key)
                            context['cache'][key] = copy.deepcopy(result)
                        except Blocked:
                            raise
                        except Exception as exc:
                            raise Blocked('validator_error', name, rule['id'], f"Validator failed ({type(exc).__name__})") from None
                    self._evidence(name, rule, context['cache'][key])
                    checked.append((rule, key))
            self._target_unchanged(name, arguments, checked)
            result = invoke() if invoke is not None else tool.handler(**arguments)
            if inspect.isawaitable(result):
                result = await result
            self._event(context, name, 'shadow' if self.mode == 'shadow' else 'forwarded')
            return result
        except Blocked as exc:
            self._event(context, name, 'blocked', exc.rule, exc.code)
            raise


def rewire(tools, plan, *, validators, **options):
    return Middleware(tools, plan, validators=validators, **options)
