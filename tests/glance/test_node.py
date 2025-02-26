from aix360.algorithms.glance.counterfactual_tree.node import Node
import pandas as pd
import unittest

class TestNodeClass(unittest.TestCase):

    def setUp(self):
        """Set up some common test cases."""
        # Create a root node and two children for testing.
        self.root = Node(split_feature="feature1", actions=[{"action": "A"}], effectiveness=5, cost=10, size=100)
        self.child1 = Node(split_feature="feature2", actions=[{"action": "B"}], effectiveness=3, cost=5, size=50)
        self.child2 = Node(split_feature="feature3", actions=[{"action": "C"}], effectiveness=2, cost=8, size=25)

    def test_node_initialization(self):
        """Test if the node initializes correctly with given parameters."""
        node = Node(split_feature="feature1", actions=[{"action": "A"}], effectiveness=5, cost=10, size=100)
        self.assertEqual(node.split_feature, "feature1")
        self.assertEqual(node.actions, [{"action": "A"}])
        self.assertEqual(node.effectiveness, 5)
        self.assertEqual(node.cost, 10)
        self.assertEqual(node.size, 100)
        self.assertEqual(node.children, {})

    def test_add_child(self):
        """Test if the add_child method works as expected."""
        subgroup1 = [1, 2]
        subgroup2 = [3, 4]
        
        # Add children to the root node.
        self.root.add_child(subgroup1, self.child1)
        self.root.add_child(subgroup2, self.child2)
        
        # Check if the children were added correctly.
        self.assertIn(tuple(subgroup1), self.root.children)
        self.assertIn(tuple(subgroup2), self.root.children)
        self.assertEqual(self.root.children[tuple(subgroup1)], self.child1)
        self.assertEqual(self.root.children[tuple(subgroup2)], self.child2)

    def test_return_leafs_actions(self):
        """Test if the return_leafs_actions method returns all actions from leaf nodes."""
        # Add children to root node.
        self.root.add_child([1, 2], self.child1)
        self.root.add_child([3, 4], self.child2)
        
        # Since both child1 and child2 are leaf nodes, their actions should be returned.
        leaf_actions = self.root.return_leafs_actions()
        self.assertEqual(len(leaf_actions), 2)
        self.assertIn({"action": "B"}, leaf_actions)
        self.assertIn({"action": "C"}, leaf_actions)

    def test_return_leafs_actions_with_nested_tree(self):
        """Test if return_leafs_actions works with a nested tree structure."""
        child3 = Node(split_feature="feature4", actions=[{"action": "D"}], effectiveness=1, cost=3, size=15)
        # Add child1 as a child of root, and child3 as a child of child1.
        self.child1.add_child([5, 6], child3)
        self.root.add_child([1, 2], self.child1)
        
        # Now, child3 is the only leaf node.
        leaf_actions = self.root.return_leafs_actions()
        self.assertEqual(leaf_actions, [{"action": "D"}])

class TestNodeGraphClass(unittest.TestCase):

    def setUp(self):
        """Set up some common test cases."""
        # Create a root node and two children for testing.
        self.root = Node(
            split_feature="feature1",
            actions=[pd.Series({"action": "A"})],  # Use Pandas Series
            effectiveness=5,
            cost=10,
            size=100
        )
        self.child1 = Node(
            split_feature="feature2",
            actions=[pd.Series({"action": "B"})],  # Use Pandas Series
            effectiveness=3,
            cost=5,
            size=50
        )
        self.child2 = Node(
            split_feature="feature3",
            actions=[pd.Series({"action": "C"})],  # Use Pandas Series
            effectiveness=2,
            cost=8,
            size=25
        )

    def test_to_igraph_structure(self):
        """Test if the iGraph object is correctly created from the node structure."""
        # Add children to the root node.
        self.root.add_child([1, 2], self.child1)
        self.root.add_child([3, 4], self.child2)

        # Convert the node tree to iGraph.
        g = self.root.to_igraph()

        # Check if the correct number of vertices is created.
        self.assertEqual(len(g.vs), 3)  # 1 root node + 2 child nodes.

        # Check if the correct number of edges is created.
        self.assertEqual(len(g.es), 2)  # 2 edges from root to its children.

        # Check the vertex labels.
        vertex_labels = g.vs["label"]
        self.assertTrue(all("action" in label for label in vertex_labels))

        # Check the edge labels.
        edge_labels = g.es["label"]
        self.assertIn("feature1", edge_labels[0])  # Root should have split on 'feature1'.

        # Check the labels of the edges.
        edge_labels = g.es["label"]
        self.assertEqual(edge_labels[0], "feature1 (1, 2)")
        self.assertEqual(edge_labels[1], "feature1 (3, 4)")

    def test_pre_order_traversal_node_ids(self):
        """Test if pre-order traversal correctly assigns node IDs."""
        # Add children to the root node.
        self.root.add_child([1, 2], self.child1)
        self.root.add_child([3, 4], self.child2)

        # Traverse the tree and assign IDs.
        self.root.to_igraph()

        # Check if the nodes were assigned unique IDs in pre-order fashion.
        self.assertEqual(self.root.id, 0)
        self.assertEqual(self.child1.id, 1)
        self.assertEqual(self.child2.id, 2)





if __name__ == "__main__":
    unittest.main()
