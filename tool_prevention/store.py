"""Durable approval/execution journal. Raw capability tokens are never stored."""
import json
import hashlib
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from .contracts import CompatibilityError,Denied,canonical


class Store:
    def __init__(self,path,clock):
        self.path=Path(path);self.clock=clock;self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.transaction() as db:
            db.executescript('''
             CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY,customer TEXT NOT NULL);
             CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
             CREATE TABLE IF NOT EXISTS requests(id TEXT PRIMARY KEY,session TEXT NOT NULL,customer TEXT NOT NULL,
                 plan TEXT NOT NULL,binding_hash TEXT NOT NULL,expires REAL NOT NULL,state TEXT NOT NULL,token_hash TEXT UNIQUE);
             CREATE TABLE IF NOT EXISTS operations(id TEXT PRIMARY KEY,session TEXT NOT NULL,operation_id TEXT NOT NULL,
                 action_hash TEXT NOT NULL,request_id TEXT NOT NULL,state TEXT NOT NULL,result TEXT,
                 UNIQUE(session,operation_id));
             CREATE TABLE IF NOT EXISTS obligations(session TEXT NOT NULL,action_hash TEXT NOT NULL,PRIMARY KEY(session,action_hash));
             CREATE TABLE IF NOT EXISTS audit(seq INTEGER PRIMARY KEY AUTOINCREMENT,t REAL NOT NULL,session TEXT,
                 tool TEXT NOT NULL,args TEXT NOT NULL,decision TEXT NOT NULL,detail TEXT NOT NULL);
            ''')

    def bind_backend(self,identity):
        if not isinstance(identity,str) or not identity:
            raise CompatibilityError('Adapter must provide a durable nonempty backend identity')
        with self.transaction() as db:
            db.execute("INSERT OR IGNORE INTO metadata VALUES ('backend_identity',?)",(identity,))
            existing=db.execute("SELECT value FROM metadata WHERE key='backend_identity'").fetchone()[0]
            if existing!=identity:raise CompatibilityError('Grant journal belongs to a different backend; use its original backend or a new journal')

    @contextmanager
    def transaction(self):
        db=sqlite3.connect(self.path,timeout=30,isolation_level=None)
        db.row_factory=sqlite3.Row
        try:
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except BaseException:
            db.rollback();raise
        finally:db.close()

    def register(self,context):
        context.validate()
        with self.transaction() as db:
            db.execute('INSERT OR IGNORE INTO sessions VALUES (?,?)',(context.session_id,context.customer_id))
            row=db.execute('SELECT customer FROM sessions WHERE id=?',(context.session_id,)).fetchone()
            if row['customer']!=context.customer_id:raise Denied('session_conflict','Session belongs to another principal')

    def event(self,context,tool,args,decision,detail):
        with self.transaction() as db:
            hashes={r[0] for r in db.execute('SELECT token_hash FROM requests WHERE token_hash IS NOT NULL')}
            def redact(value):
                if isinstance(value,dict):
                    return {redact(k):('[REDACTED]' if k.lower() in ('approval_token','api_key','authorization','token') else redact(v)) for k,v in value.items()}
                if isinstance(value,list):return [redact(v) for v in value]
                if isinstance(value,str) and hashlib.sha256(value.encode()).hexdigest() in hashes:return '[REDACTED]'
                return value
            db.execute('INSERT INTO audit(t,session,tool,args,decision,detail) VALUES (?,?,?,?,?,?)',
                (self.clock(),context.session_id if context else None,redact(tool),canonical(redact(args)),decision,canonical(redact(detail))))

    def audit(self,session=None):
        with self.transaction() as db:
            rows=db.execute('SELECT * FROM audit'+(' WHERE session=?' if session else '')+' ORDER BY seq',
                            (session,) if session else ()).fetchall()
        return [{**dict(r),'args':json.loads(r['args']),'detail':json.loads(r['detail'])} for r in rows]
