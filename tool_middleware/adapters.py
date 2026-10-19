"""Adapt a host's original tool dispatcher; no transport or endpoint changes.

Definitions can be neutral, OpenAI-style, or MCP-style metadata. The executor must
return the declared JSON shape (transport envelopes require a host adapter).
"""
import copy
import inspect
from .catalog import Tool, PlanError


def is_async(handler):
    return inspect.iscoroutinefunction(handler) or inspect.iscoroutinefunction(getattr(handler, '__call__', None))


def from_dispatcher(definitions, execute, *, effects=None, output_schemas=None):
    effects, output_schemas = effects or {}, output_schemas or {}
    result = []
    for original in definitions:
        definition = original.get('function', original)
        name = definition['name']
        schema = next((definition[key] for key in ('input_schema', 'inputSchema', 'parameters') if key in definition), None)
        if schema is None:
            raise PlanError(f'Missing input schema for {name}')
        output = output_schemas.get(name, definition.get('output_schema', definition.get('outputSchema')))
        if is_async(execute):
            def bind_async(tool_name):
                async def invoke(**arguments):
                    return await execute(tool_name, arguments)
                return invoke
            handler = bind_async(name)
        else:
            def bind(tool_name):
                def invoke(**arguments):
                    return execute(tool_name, arguments)
                return invoke
            handler = bind(name)
        result.append(Tool(name, definition.get('description', ''), copy.deepcopy(schema), handler,
                           copy.deepcopy(output), effects.get(name, 'unknown')))
    return result
