"""Executable local retail reference backend, not a live tau-bench service."""
import copy
import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from .contracts import Denied,canonical

ADDRESS=['address1','address2','city','country','state','zip']
READ_FIELDS={
 'find_user_id_by_email':{'email':'string'},
 'find_user_id_by_name_zip':{'first_name':'string','last_name':'string','zip':'string'},
 'list_all_product_types':{},'get_product_details':{'product_id':'string'},
 'get_user_details':{'user_id':'string'},'get_order_details':{'order_id':'string'},
}
WRITE_FIELDS={
 'cancel_pending_order':{'order_id':'string','reason':'string'},
 'modify_pending_order_address':{'order_id':'string',**{k:'string' for k in ADDRESS}},
 'modify_user_address':{'user_id':'string',**{k:'string' for k in ADDRESS}},
 'modify_pending_order_items':{'order_id':'string','item_ids':'strings','new_item_ids':'strings','payment_method_id':'string'},
 'modify_pending_order_payment':{'order_id':'string','payment_method_id':'string'},
 'return_delivered_order_items':{'order_id':'string','item_ids':'strings','payment_method_id':'string'},
 'exchange_delivered_order_items':{'order_id':'string','item_ids':'strings','new_item_ids':'strings','payment_method_id':'string'},
}


def schema(fields):
    return {'type':'object','properties':{k:({'type':'array','items':{'type':'string'},'minItems':1}
             if t=='strings' else {'type':t}) for k,t in fields.items()},
            'required':list(fields),'additionalProperties':False}


def sample_data():
    return {
      'user:alice':{'user_id':'alice','email':'alice@example.test','first_name':'Alice','last_name':'Example','zip':'12345',
                    'payment_methods':{'card-alice':{'type':'credit_card'},'gift-alice':{'type':'gift_card','balance_cents':10000}}},
      'user:bob':{'user_id':'bob','email':'bob@example.test','first_name':'Bob','last_name':'Example','zip':'54321',
                  'payment_methods':{'card-bob':{'type':'credit_card'}}},
      'product:shirt':{'product_id':'shirt','name':'T-Shirt','variants':{
          'shirt-a':{'item_id':'shirt-a','product_id':'shirt','price_cents':2000,'available':True},
          'shirt-b':{'item_id':'shirt-b','product_id':'shirt','price_cents':2500,'available':True},
          'shirt-c':{'item_id':'shirt-c','product_id':'shirt','price_cents':2200,'available':False}}},
      'product:shoe':{'product_id':'shoe','name':'Shoes','variants':{
          'shoe-a':{'item_id':'shoe-a','product_id':'shoe','price_cents':1000,'available':True}}},
      'order:pending-alice':{'order_id':'pending-alice','user_id':'alice','status':'pending','items':[
          {'item_id':'shirt-a','product_id':'shirt','price_cents':2000},
          {'item_id':'shoe-a','product_id':'shoe','price_cents':1000}], 'payment_method_id':'card-alice','operations':[]},
      'order:delivered-alice':{'order_id':'delivered-alice','user_id':'alice','status':'delivered','items':[
          {'item_id':'shirt-a','product_id':'shirt','price_cents':2000}], 'payment_method_id':'card-alice','operations':[]},
      'order:pending-bob':{'order_id':'pending-bob','user_id':'bob','status':'pending','items':[
          {'item_id':'shirt-a','product_id':'shirt','price_cents':2000}], 'payment_method_id':'card-bob','operations':[]},
    }


class RetailSystem:
    capabilities=frozenset({'conditional_writes','durable_idempotency'})

    def __init__(self,path): self.path=Path(path)

    @classmethod
    def create(cls,path,data=None):
        obj=cls(path);obj.path.parent.mkdir(parents=True,exist_ok=True)
        with obj.connect() as db:
            db.executescript('''CREATE TABLE IF NOT EXISTS resources(key TEXT PRIMARY KEY,value TEXT NOT NULL,version INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS effects(operation_id TEXT PRIMARY KEY,action_hash TEXT NOT NULL,result TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);''')
            db.execute("INSERT OR IGNORE INTO metadata VALUES ('backend_identity',?)",('retail:'+uuid.uuid4().hex,))
            for key,value in (data if data is not None else sample_data()).items():
                db.execute('INSERT OR IGNORE INTO resources VALUES (?,?,1)',(key,canonical(value)))
        return obj

    def backend_identity(self):
        with self.connect() as db:
            return db.execute("SELECT value FROM metadata WHERE key='backend_identity'").fetchone()[0]

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.path,timeout=30,isolation_level='IMMEDIATE')
        try:
            with db:
                yield db
        finally:
            db.close()

    def tool_specs(self):
        return {name:{'name':name,'description':name.replace('_',' '),'parameters':schema(fields),
                       'mutates':name in WRITE_FIELDS} for name,fields in {**READ_FIELDS,**WRITE_FIELDS}.items()}

    def snapshot(self):
        with self.connect() as db:
            rows=db.execute('SELECT key,value,version FROM resources').fetchall()
        return {'resources':{k:json.loads(v) for k,v,_ in rows},'versions':{k:n for k,_,n in rows}}

    def read_resource(self,kind,identity):
        return copy.deepcopy(self.snapshot()['resources'][kind+':'+identity])

    def read(self,context,tool,args):
        snapshot=self.snapshot()['resources']
        user=snapshot.get('user:'+context.customer_id)
        if not user: raise Denied('unknown_customer','Unknown trusted customer')
        if tool=='get_user_details':
            if args['user_id']!=context.customer_id: raise Denied('ownership','Cannot read another customer')
            return user
        if tool=='get_order_details':
            order=snapshot.get('order:'+args['order_id'])
            if not order or order['user_id']!=context.customer_id: raise Denied('ownership','Cannot read another customer order')
            return order
        if tool=='get_product_details':
            if 'product:'+args['product_id'] not in snapshot: raise Denied('not_found','Unknown product')
            return snapshot['product:'+args['product_id']]
        if tool=='list_all_product_types':
            return {v['name']:v['product_id'] for k,v in snapshot.items() if k.startswith('product:')}
        if tool=='find_user_id_by_email':
            if args['email']!=user['email']: raise Denied('ownership','Identity lookup does not match trusted session')
            return user['user_id']
        if tool=='find_user_id_by_name_zip':
            if any(args[k]!=user[k] for k in ('first_name','last_name','zip')):
                raise Denied('ownership','Identity lookup does not match trusted session')
            return user['user_id']
        raise Denied('unknown_tool','Unknown read tool')

    def execute_atomic(self,plan,operation_id):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            cached=db.execute('SELECT action_hash,result FROM effects WHERE operation_id=?',(operation_id,)).fetchone()
            if cached:
                if cached[0]!=plan.action_hash: raise Denied('operation_conflict','Backend operation ID reused for another action')
                return json.loads(cached[1])
            for key,version in plan.versions.items():
                row=db.execute('SELECT version FROM resources WHERE key=?',(key,)).fetchone()
                if not row or row[0]!=version: raise Denied('stale_state','Resource changed after approval')
            for key,value in plan.effects.items():
                cursor=db.execute('UPDATE resources SET value=?,version=version+1 WHERE key=? AND version=?',
                                  (canonical(value),key,plan.versions[key]))
                if cursor.rowcount!=1: raise Denied('stale_state','Conditional mutation failed')
            result={'tool':plan.tool,'arguments':plan.arguments,'quote':plan.quote,'updated_resources':list(plan.effects),
                    'operation_id':operation_id}
            db.execute('INSERT INTO effects VALUES (?,?,?)',(operation_id,plan.action_hash,canonical(result)))
            return result

    def lookup_execution(self,operation_id):
        with self.connect() as db:
            row=db.execute('SELECT action_hash,result FROM effects WHERE operation_id=?',(operation_id,)).fetchone()
        return {'status':'succeeded','action_hash':row[0],'result':json.loads(row[1])} if row else {'status':'unknown'}

    def side_effect_count(self):
        with self.connect() as db:return db.execute('SELECT COUNT(*) FROM effects').fetchone()[0]
