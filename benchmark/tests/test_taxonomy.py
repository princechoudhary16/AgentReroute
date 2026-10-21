import unittest

from benchmark import taxonomy


class TaxonomyTest(unittest.TestCase):
    def test_seven_modes_in_order(self):
        self.assertEqual(len(taxonomy.MODES), 7)
        self.assertEqual(taxonomy.MODES[0], 'instruction_adherence')
        self.assertIn('out_of_scope', taxonomy.LABELS)

    def test_aliases(self):
        self.assertEqual(taxonomy.mode('Cycles'), 'cycles')
        self.assertEqual(taxonomy.mode(4), 'tool_failure')
        self.assertEqual(taxonomy.mode('Missing Details in Final Output'), 'missing_details')
        self.assertEqual(taxonomy.mode('abstain'), 'inconclusive')
        with self.assertRaises(ValueError):
            taxonomy.mode('hallucination-ish')

    def test_agentrx_mapping_complete_both_ways(self):
        for category in ('Instruction/PlanAdherenceFailure', 'InventionOfNewInformation', 'InvalidInvocation',
                         'MisinterpretationOfToolOutput', 'IntentPlanMisalignment', 'UnderspecifiedUserIntent',
                         'IntentNotSupported', 'GuardrailsTriggered', 'SystemFailure', 'Inconclusive'):
            mode, fit = taxonomy.from_agentrx(category)
            self.assertIn(mode, taxonomy.LABELS + ('inconclusive',))
            self.assertIn(fit, ('exact', 'approximate', 'none'))
        for mode in taxonomy.MODES:
            self.assertTrue(taxonomy.to_agentrx(mode))

    def test_mast_mapping(self):
        self.assertEqual(taxonomy.from_mast('FM-1.3 Step repetition')[0], 'cycles')
        self.assertEqual(taxonomy.from_mast('FM-2.3')[0], 'drift')


if __name__ == '__main__':
    unittest.main()
