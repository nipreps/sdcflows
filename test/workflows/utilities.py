'''  Class and utilities for testing the workflows module '''

import unittest
from networkx.exception import NetworkXUnfeasible

from nipype.pipeline import engine
from nipype.interfaces import utility

class TestWorkflow(unittest.TestCase):
    ''' Subclass for test within the workflow module.
    invoke tests with ``python -m unittest discover test'''

    def assertIsAlmostExpectedWorkflow(self, expected_name, expected_interfaces,
                                       expected_inputs, expected_outputs,
                                       actual):
        ''' somewhat hacky way to confirm workflows are as expected, but with low confidence '''
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

    def assert_circular(self, workflow, circular_connections):
        ''' check key paths in workflow by specifying some connections that should induce
        circular paths, which trips a NetworkX error.
        circular_connections is a list of tuples:
            [('from_node_name', 'to_node_name', ('from_node.output_field','to_node.input_field'))]
        '''

        for from_node, to_node, fields in circular_connections:
            from_node = workflow.get_node(from_node)
            to_node = workflow.get_node(to_node)
            workflow.connect([(from_node, to_node, fields)])

            self.assertRaises(NetworkXUnfeasible, workflow.write_graph)

            workflow.disconnect([(from_node, to_node, fields)])

    def assert_inputs_set(self, workflow, mandatory_inputs):
        ''' check that all inputs in the mandatory_inputs list are already set by attempting
        to connect an arbitrary output to each of the inputs, which trips an error.
        mandatory_inputs is a dict:
            {'node_name': ['mandatory', 'input', 'fields']}'''

        for node_name, fields in mandatory_inputs.items():
            dummy_node = engine.Node(utility.IdentityInterface(fields=['dummy']), name='DummyNode')

            node = workflow.get_node(node_name)
            for field in fields:
                with self.assertRaises(Exception):
                    workflow.connect([(dummy_node, node, [('dummy', field)])])
