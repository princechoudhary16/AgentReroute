"""Pre-execution registry rewiring and a separate trusted approval interface."""
import copy
import hashlib
import json
import secrets
import time
from .contracts import Context,Denied,action_hash,fingerprint
from .policy import compile_policy,default_policy
from .store import Store


def _token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest() if isinstance(token,str) else ''


def _operation_key(context,identifier):
    return fingerprint({'session':context.session_id,'customer':context.customer_id,'operation':identifier})


def _stored_result(row):
    result=json.loads(row['result']) if row['result'] else None
    if row['state']=='succeeded':return {'status':'succeeded','cached':True,'result':result}
    if row['state']=='rejected':return {'status':'denied',**(result or {'code':'backend_rejected'})}
    return {'status':'indeterminate','code':'reconciliation_required','operation_id':row['operation_id']}


class ProtectedSystem:
    def __init__(self,system,policy,store):
        self._system,self.policy,self._store=system,policy,store
        self.admin=ApprovalAdmin(self)
        self.coverage=policy.coverage

    def request_approval(self,context,tool,arguments):
        try:
            arguments=copy.deepcopy(arguments);self._store.register(context)
            plan=self.policy.plan(context,tool,arguments)
            request_id=secrets.token_urlsafe(24)
            expires=self._store.clock()+self.policy.ttl
            with self._store.transaction() as db:
                db.execute('INSERT INTO requests VALUES (?,?,?,?,?,?,?,NULL)',(request_id,context.session_id,
                           context.customer_id,json.dumps(plan.to_dict()),plan.binding_hash,expires,'pending'))
            result={'status':'approval_pending','request_id':request_id,'expires_at':expires,
                    'action':{'tool':tool,'arguments':arguments,'quote':plan.quote}}
            self._store.event(context,tool,arguments,'approval_requested',{'request_id':request_id,'quote':plan.quote})
            return result
        except Denied as exc:
            return self._denied(context,tool,arguments,exc)

    def call(self,context,tool,arguments,operation_id=None,approval_token=None):
        arguments=copy.deepcopy(arguments)
        try:
            self._store.register(context)
            if tool=='request_action_approval':
                if not isinstance(arguments,dict) or set(arguments)!={'tool_name','arguments'}:
                    raise Denied('invalid_arguments','Approval request needs tool_name and arguments')
                return self.request_approval(context,arguments['tool_name'],arguments['arguments'])
            if tool=='complete_task':
                if arguments:raise Denied('invalid_arguments','Completion takes no arguments')
                return self.complete_task(context)
            self.policy.validate(context,tool,arguments)
            if not self.policy.specs[tool]['mutates']:
                result=self._system.read(context,tool,arguments)
                self._store.event(context,tool,arguments,'succeeded',{'result':result})
                return {'status':'succeeded','result':result}
            if not isinstance(operation_id,str) or not operation_id or len(operation_id)>256:
                raise Denied('operation_id_required','Mutations need a stable operation_id')
            action=action_hash(tool,arguments,context.customer_id,context.session_id,self.policy.version)
            operation_key=_operation_key(context,operation_id)
            with self._store.transaction() as db:
                previous=db.execute('SELECT * FROM operations WHERE id=?',(operation_key,)).fetchone()
            if previous:
                if previous['action_hash']!=action:raise Denied('operation_conflict','Operation ID cannot identify different actions')
                return self._cached(context,tool,arguments,previous)
            if not isinstance(approval_token,str) or not approval_token:
                raise Denied('approval_required','Request exact-action approval before mutation')
            plan=self.policy.plan(context,tool,arguments)
            with self._store.transaction() as db:
                previous=db.execute('SELECT * FROM operations WHERE id=?',(operation_key,)).fetchone()
                if previous:
                    if previous['action_hash']!=action:raise Denied('operation_conflict','Operation ID cannot identify different actions')
                else:
                    grant=db.execute('SELECT * FROM requests WHERE token_hash=?',(_token_hash(approval_token),)).fetchone()
                    if not grant or grant['state']!='approved':raise Denied('invalid_approval','Approval is unknown, denied, or already used')
                    if grant['session']!=context.session_id or grant['customer']!=context.customer_id:
                        raise Denied('approval_scope','Approval belongs to another session/customer')
                    if grant['expires']<=self._store.clock():raise Denied('approval_expired','Approval expired')
                    if grant['binding_hash']!=plan.binding_hash:raise Denied('approval_mismatch','Action, quote, policy, or resource version changed after approval')
                    db.execute("UPDATE requests SET state='consumed' WHERE id=?",(grant['id'],))
                    db.execute('INSERT INTO operations VALUES (?,?,?,?,?,?,NULL)',(operation_key,context.session_id,
                               operation_id,action,grant['id'],'running'))
            if previous:return self._cached(context,tool,arguments,previous)
            try:
                result=self._system.execute_atomic(plan,operation_key)
            except Denied as exc:
                value={'code':exc.code,'message':str(exc)}
                self._finish(operation_key,'rejected',value)
                return self._denied(context,tool,arguments,exc)
            except Exception as exc:
                # The backend may have committed. Never release its grant or retry blindly.
                self._finish(operation_key,'indeterminate',{'error_type':type(exc).__name__})
                self._store.event(context,tool,arguments,'indeterminate',{'operation_id':operation_id,'error_type':type(exc).__name__})
                return {'status':'indeterminate','code':'reconciliation_required','operation_id':operation_id}
            self._finish(operation_key,'succeeded',result)
            self._store.event(context,tool,arguments,'succeeded',{'operation_id':operation_id,'result':result})
            return {'status':'succeeded','cached':False,'result':result}
        except Denied as exc:
            return self._denied(context,tool,arguments,exc)

    def _cached(self,context,tool,arguments,row):
        result=_stored_result(row)
        self._store.event(context,tool,arguments,'cached',{'operation_id':row['operation_id'],'outcome':result['status']})
        return result

    def _finish(self,key,state,result):
        with self._store.transaction() as db:
            # Reconciliation may confirm success while the original caller waits.
            db.execute("UPDATE operations SET state=?,result=? WHERE id=? AND state!='succeeded'",(state,json.dumps(result),key))

    def _denied(self,context,tool,args,exc):
        self._store.event(context,tool,args,'denied',{'code':exc.code,'message':str(exc)})
        return {'status':'denied','code':exc.code,'message':str(exc)}

    def require_action(self,context,tool,args):
        """Trusted host-only finite obligation; natural-language goals are not inferred."""
        self._store.register(context);self.policy.validate(context,tool,args)
        if not self.policy.specs[tool]['mutates']:raise Denied('invalid_obligation','Completion obligations must name mutations')
        identity=action_hash(tool,args,context.customer_id,context.session_id,self.policy.version)
        with self._store.transaction() as db:db.execute('INSERT OR IGNORE INTO obligations VALUES (?,?)',(context.session_id,identity))

    def complete_task(self,context):
        try:
            self._store.register(context)
            with self._store.transaction() as db:
                rows=db.execute("SELECT o.action_hash FROM obligations o WHERE o.session=? AND NOT EXISTS "
                    "(SELECT 1 FROM operations e WHERE e.session=o.session AND e.action_hash=o.action_hash AND e.state='succeeded')",(context.session_id,)).fetchall()
            if rows:raise Denied('incomplete_obligations','Explicit host-required actions remain unfinished')
            self._store.event(context,'complete_task',{},'completed',{})
            return {'status':'completed'}
        except Denied as exc:return self._denied(context,'complete_task',{},exc)

    def tool_definitions(self):
        specs=copy.deepcopy(self.policy.specs)
        for spec in specs.values():
            if spec['mutates']:
                for name in ('operation_id','approval_token'):
                    spec['parameters']['properties'][name]={'type':'string'}
                    spec['parameters']['required'].append(name)
        definitions=[{'name':s['name'],'description':s['description'],'parameters':s['parameters']} for s in specs.values()]
        definitions.extend([
            {'name':'request_action_approval','description':'Request approval from the trusted user; does not grant approval.',
             'parameters':{'type':'object','properties':{'tool_name':{'type':'string'},'arguments':{'type':'object'}},
                           'required':['tool_name','arguments'],'additionalProperties':False}},
            {'name':'complete_task','description':'Finish only after explicit host obligations are satisfied.',
             'parameters':{'type':'object','properties':{},'required':[],'additionalProperties':False}}])
        return definitions

    def tools(self,context):
        self._store.register(context)
        def bind(name):
            def invoke(**arguments):
                operation_id=arguments.pop('operation_id',None);token=arguments.pop('approval_token',None)
                return self.call(context,name,arguments,operation_id,token)
            return invoke
        return {d['name']:bind(d['name']) for d in self.tool_definitions()}

    def audit(self):return self._store.audit()

    def export_trace(self,context):
        steps=[]
        for event in self._store.audit(context.session_id):
            index=len(steps)+1
            steps.append({'index':index,'substeps':[{'sub_index':1,'role':'assistant','content':'',
                'tool_calls':[{'id':str(event['seq']),'name':event['tool'],'arguments':event['args']}]}]})
            steps.append({'index':index+1,'substeps':[{'sub_index':1,'role':'tool','name':event['tool'],
                'content':json.dumps({'status':event['decision'],**event['detail']})}]})
        return {'trajectory_id':context.session_id,'domain':'tau','instruction':'Protected retail tool execution','steps':steps}


class ApprovalAdmin:
    """Trusted host interface. Never include it in the agent-visible registry."""
    def __init__(self,gateway):self._gateway=gateway

    def preview(self,request_id):
        with self._gateway._store.transaction() as db:
            row=db.execute('SELECT * FROM requests WHERE id=?',(request_id,)).fetchone()
        if not row:raise Denied('unknown_request','Approval request does not exist')
        return {'request_id':request_id,'state':row['state'],'expires_at':row['expires'],'action':json.loads(row['plan'])}

    def approve(self,request_id):
        gateway=self._gateway
        token=secrets.token_urlsafe(32)
        with gateway._store.transaction() as db:
            row=db.execute('SELECT * FROM requests WHERE id=?',(request_id,)).fetchone()
            if not row or row['state']!='pending':raise Denied('approval_state','Only pending requests can be approved')
            plan=json.loads(row['plan'])
            if row['expires']<=gateway._store.clock():raise Denied('approval_expired','Request expired; request a new quote')
            if plan['policy_version']!=gateway.policy.version:raise Denied('policy_changed','Policy changed; request new approval')
            db.execute("UPDATE requests SET state='approved',token_hash=? WHERE id=?",(_token_hash(token),request_id))
        context=Context(row['session'],row['customer'])
        gateway._store.event(context,plan['tool'],plan['arguments'],'approved',{'request_id':request_id,'quote':plan['quote']})
        return {'status':'approved','request_id':request_id,'approval_token':token,'action':plan}

    def deny(self,request_id):
        with self._gateway._store.transaction() as db:
            row=db.execute('SELECT * FROM requests WHERE id=?',(request_id,)).fetchone()
            if not row or row['state'] not in ('pending','approved'):raise Denied('approval_state','Request cannot be denied in current state')
            db.execute("UPDATE requests SET state='denied' WHERE id=?",(request_id,))
        plan=json.loads(row['plan'])
        self._gateway._store.event(Context(row['session'],row['customer']),plan['tool'],plan['arguments'],
                                   'approval_denied',{'request_id':request_id})
        return {'status':'denied','request_id':request_id}

    def reconcile(self,context,operation_id):
        gateway=self._gateway;gateway._store.register(context);key=_operation_key(context,operation_id)
        with gateway._store.transaction() as db:
            row=db.execute('SELECT * FROM operations WHERE id=?',(key,)).fetchone()
        if not row:raise Denied('unknown_operation','Operation does not exist')
        if row['state']=='succeeded':return _stored_result(row)
        evidence=gateway._system.lookup_execution(key)
        if evidence.get('status')=='succeeded' and evidence.get('action_hash')==row['action_hash']:
            gateway._finish(key,'succeeded',evidence['result'])
            gateway._store.event(context,'reconcile',{},'succeeded',{'operation_id':operation_id})
            return {'status':'succeeded','cached':True,'result':evidence['result']}
        return {'status':'indeterminate','code':'reconciliation_required'}


def rewire(system,policy=None,store_path='runs/prevention/grants.db',clock=time.time):
    compiled=compile_policy(system,policy if policy is not None else default_policy())
    store=Store(store_path,clock)
    store.bind_backend(system.backend_identity())
    return ProtectedSystem(system,compiled,store)
