"""Hook for the project's planned env.step(Action(name, kwargs)) dispatcher.

Install this at the proposed-action boundary; a completed Rollout is too late.
"""
from .contracts import CompatibilityError


class ProtectedDispatcher:
    def __init__(self,protected,context):
        self.protected,self.context=protected,context
        self.registry=protected.tools(context)

    def step(self,action):
        name=getattr(action,'name',None);arguments=getattr(action,'kwargs',None)
        if not isinstance(name,str) or not isinstance(arguments,dict):
            raise CompatibilityError('Expected a proposed Action(name, kwargs), not an executed trace or rollout')
        function=self.registry.get(name)
        if function is None:
            return self.protected.call(self.context,name,arguments)
        return function(**arguments)
