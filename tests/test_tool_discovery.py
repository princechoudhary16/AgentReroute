import copy
import json
import unittest
from tool_middleware.discovery import discover, DiscoveryError
from tool_middleware import Middleware, PlanError
from tests.test_tool_middleware import shipping


class Model:
    def __init__(self,response):self.response=response;self.requests=[]
    def complete_messages(self,messages):
        self.requests.append(messages)
        return copy.deepcopy(self.response)


class ToolDiscoveryTests(unittest.TestCase):
    def test_metadata_discovery_creates_an_inspectable_plan_without_invoking_tools(self):
        tools,reference,calls,_=shipping()
        model=Model({'rules':reference['rules'],'abstentions':[]})
        plan=discover(tools,model,validators=['validate_shipment_number'])
        self.assertEqual(calls,[])
        self.assertEqual(plan['catalog_hash'],reference['catalog_hash'])
        self.assertEqual(plan['rules'],reference['rules'])
        request=json.loads(model.requests[0][1]['content'])
        self.assertEqual(request['validators'],['validate_shipment_number'])
        self.assertNotIn('handler',model.requests[0][1]['content'])
        self.assertIn('never instructions',model.requests[0][0]['content'])
        layer=Middleware(tools,plan,validators=['validate_shipment_number'])
        layer.dispatch('get_shipment_status',{'shipment_number':'SHIP-123'})
        self.assertEqual(calls,[('validate','SHIP-123'),('status','SHIP-123')])

    def test_uncertain_invented_and_incompatible_suggestions_become_explicit_abstentions(self):
        for mode in ('low_confidence','unknown_validator','missing_schema','missing_binding','malformed_fields'):
            tools,plan,calls,_=shipping();rule=plan['rules'][0]
            if mode=='low_confidence':rule['confidence']='medium'
            if mode=='unknown_validator':rule['validator']='invented'
            if mode=='missing_schema':tools[0].output_schema=None
            if mode=='missing_binding':rule['bindings']={}
            if mode=='malformed_fields':rule['target']=[]
            result=discover(tools,Model({'rules':[rule],'abstentions':[]}),validators=['validate_shipment_number'])
            with self.subTest(mode=mode):
                self.assertEqual(result['rules'],[])
                self.assertTrue(result['abstentions'])
                self.assertEqual(calls,[])

    def test_combined_cycle_is_not_activated(self):
        tools,plan,_,_=shipping()
        tools[1].effect='read_only';tools[1].output_schema=copy.deepcopy(tools[0].output_schema)
        reverse=copy.deepcopy(plan['rules'][0]);reverse.update(id='reverse',target='validate_shipment_number',validator='get_shipment_status')
        result=discover(tools,Model({'rules':[plan['rules'][0],reverse],'abstentions':[]}),validators=[t.name for t in tools])
        self.assertEqual(len(result['rules']),1)
        self.assertTrue(result['abstentions'])

    def test_bad_model_response_fails_and_never_changes_the_catalog(self):
        tools,plan,calls,_=shipping();before=[t.definition() for t in tools]
        for response in ([],{}, {'rules':'run arbitrary code','abstentions':[]}):
            with self.subTest(response=response),self.assertRaises(DiscoveryError):discover(tools,Model(response),validators=['validate_shipment_number'])
        self.assertEqual(before,[t.definition() for t in tools]);self.assertEqual(calls,[])

    def test_model_cannot_designate_an_automatic_validator(self):
        tools,plan,calls,_=shipping()
        result=discover(tools,Model({'rules':plan['rules'],'abstentions':[]}),validators=[])
        self.assertEqual(result['rules'],[]);self.assertEqual(calls,[])
