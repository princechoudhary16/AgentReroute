"""Trusted host/adapter contracts. Agent inputs never establish identity or approval."""
from dataclasses import dataclass, field, asdict
import hashlib
import json


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',', ':'),ensure_ascii=False,allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class CompatibilityError(ValueError): pass


class Denied(ValueError):
    def __init__(self,code,message):
        self.code=code
        super().__init__(message)


class UncertainExecution(RuntimeError): pass


@dataclass(frozen=True)
class Context:
    session_id: str
    customer_id: str
    authenticated: bool = True

    def validate(self):
        if not isinstance(self.session_id,str) or not self.session_id or not isinstance(self.customer_id,str) or not self.customer_id:
            raise Denied('invalid_context','Host identity and session are required')
        if self.authenticated is not True:
            raise Denied('authentication_required','Authenticate outside the agent before using these tools')


@dataclass
class ActionPlan:
    tool: str
    arguments: dict
    customer_id: str
    session_id: str
    policy_version: str
    quote: dict
    versions: dict
    effects: dict = field(default_factory=dict)

    @property
    def action_hash(self):
        return action_hash(self.tool,self.arguments,self.customer_id,self.session_id,self.policy_version)

    @property
    def binding_hash(self):
        return fingerprint(asdict(self))

    def to_dict(self): return asdict(self)


def action_hash(tool,args,customer,session,policy):
    return fingerprint({'tool':tool,'arguments':args,'customer_id':customer,'session_id':session,'policy_version':policy})
