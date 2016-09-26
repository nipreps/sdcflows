'''  Class and utilities for testing the workflows module '''

import unittest

from nipype.pipeline import engine
from nipype.interfaces import IdentityInterface

class TestWorkflow(unittest.TestCase):
    ''' Subclass for test within the workflow module.
    invoke tests with ``python -m unittest discover test'''

    def assertIsAlmostExpectedWorkflow(self, expected_name, expected_interfaces,
                                       expected_inputs, expected_outputs,
                                       actual):
        self.assertIsInstance(actual, engine.Workflow)
        self.assertEqual(expected_name, actual.name)

        # assert it has the same nodes
        actual_nodes = [actual.get_node(name)
                        for name in actual.list_node_names()]
        actual_interfaces = [node.interface.__class__.__name__
                             for node in actual_nodes]

        # assert lists equal
        self.assertIsSubsetOfList(expected_interfaces, actual_interfaces)
        self.assertIsSubsetOfList(actual_interfaces, expected_interfaces)

        # assert expected inputs, outputs exist
        actual_inputs, actual_outputs = self.get_inputs_outputs(actual_nodes)

        self.assertIsSubsetOfList(expected_outputs, actual_outputs)
        self.assertIsSubsetOfList(expected_inputs, actual_inputs)

    def assertIsSubsetOfList(self, expecteds, actuals):
        for expected in expecteds:
            self.assertIn(expected, actuals)

    def get_inputs_outputs(self, nodes):
        def get_io_names(pre, ios):
            return [pre + str(io[0]) for io in ios]

        actual_inputs = []
        actual_outputs = []
        node_tuples = [(node.name, node.inputs.items(), node.outputs.items())
                       for node in nodes]
        for name, inputs, outputs in node_tuples:
            pre = str(name) + "."
            actual_inputs += get_io_names(pre, inputs)

            pre = pre if pre[0:-1] != 'inputnode' else ""
            actual_outputs += get_io_names(pre, outputs)

        return actual_inputs, actual_outputs

def stub_node_factory(*args, **kwargs):
    ''' For use with mock.patch.
    Stubs out a "Node" to modularize testing of workflow creation and validation '''
    return engine.Node(IdentityInterface(fields=['inputnode', 'outputnode']), name=kwargs['name'])
