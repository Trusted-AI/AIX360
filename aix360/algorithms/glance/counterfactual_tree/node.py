class Node:
    """
    A class representing a node in a decision tree structure.

    Each node can have child nodes, actions associated with it, and metrics 
    such as effectiveness and cost. This class provides methods to add child 
    nodes, retrieve actions from leaf nodes, and visualize the tree structure.

    Attributes:
    ----------
    split_feature : str or None
        The feature used to split the data at this node. Default is None.
        
    actions : list or None
        A list of actions associated with this node. Default is None.
        
    effectiveness : float
        The effectiveness of the actions taken at this node. Default is 0.
        
    cost : float
        The total cost associated with the actions at this node. Default is 0.
        
    size : int
        The number of instances or data points at this node. Default is 0.
        
    children : dict
        A dictionary mapping from subgroup values to child nodes.
        
    Methods:
    -------
    add_child(subgroup, child_node):
        Adds a child node to this node.
    
    return_leafs_actions():
        Returns all actions from the leaf nodes in the subtree rooted at this node.
        
    to_igraph(numeric_features=[]):
        Converts the tree structure to an igraph object for visualization.
        
    display_igraph_jupyter(numeric_features=[]):
        Displays the tree structure in a Jupyter notebook using matplotlib and igraph.
    """

    def __init__(
        self,
        split_feature=None,
        actions=None,
        effectiveness=0,
        cost=0,
        size=0,
    ):
        """
        Initializes a new Node instance.

        Parameters:
        ----------
        split_feature : str or None
            The feature used to split the data at this node. Default is None.
            
        actions : list or None
            A list of actions associated with this node. Default is None.
            
        effectiveness : float
            The effectiveness of the actions taken at this node. Default is 0.
            
        cost : float
            The total cost associated with the actions at this node. Default is 0.
            
        size : int
            The number of instances or data points at this node. Default is 0.
        """
        self.split_feature = split_feature
        self.effectiveness = effectiveness
        self.cost = cost
        self.size = size
        self.children = {}
        self.actions = actions

    def add_child(self, subgroup, child_node):
        """
        Adds a child node to this node.

        Parameters:
        ----------
        subgroup : any
            The value associated with the child node.
            
        child_node : Node
            The child node to be added.
        """
        print()
        self.children[tuple(subgroup)] = child_node

    def return_leafs_actions(self):
        """
        Returns all actions from the leaf nodes in the subtree rooted at this node.

        Returns:
        -------
        list
            A flattened list of actions from the leaf nodes.
        """
        cfs_list = []

        def find_leafs_actions(node):
            if node.children == {}:
                cfs_list.append(node.actions)
            else:
                for child_node in node.children.values():
                    find_leafs_actions(child_node)

        find_leafs_actions(self)
        return [action for sublist in cfs_list for action in sublist]

    
    def to_igraph(self, numeric_features=[]):
        """
        Converts the tree structure to an igraph object for visualization.

        Parameters:
        ----------
        numeric_features : list
            A list of numeric feature names used for processing node labels.

        Returns:
        -------
        ig.Graph
            An igraph object representing the tree structure.
        """
        import igraph as ig

        def pre_order(node, timer, reg):
            node.id = timer
            reg[timer] = node
            timer += 1
            for _value, child_node in node.children.items():
                timer = pre_order(child_node, timer=timer, reg=reg)
            return timer

        node_registry = dict()
        n_nodes = pre_order(self, timer=0, reg=node_registry)

        def pre_order_2(node):
            max_value = 0 
            for value, child_node in node.children.items():
                if node.split_feature in numeric_features:
                    if max(value) > max_value:  
                        max_value = max(value)
                        max_list = child_node
            for value, child_node in node.children.items():
                if node.split_feature in numeric_features:
                    if child_node != max_list:
                        val = max(value)
            for value, child_node in node.children.items():
                child_node.data_feat = node.split_feature
                if node.split_feature in numeric_features:
                    if child_node == max_list:
                        child_node.data_val = f"> {val}"
                    else:   
                        child_node.data_val = f"<= {val}"
                else:
                    child_node.data_val = value if len(value) > 1 else value[0]
                pre_order_2(child_node)

        pre_order_2(self)
        self.data_feat = "all"
        self.data_val = "-"

        graph = ig.Graph(directed=True)
        
        def add_nodes(node):
            size = node.size
            num_flipped = node.effectiveness
            cost_sum = node.cost
            eff = num_flipped / size
            actions = [action[action != "-"].to_dict() for action in node.actions]
            actions_ = []

            for action in actions:
                action_copy = action
                for k, v in action_copy.items():
                    if k in numeric_features:
                        action_copy[k] = round(v, 3)
                actions_.append(action_copy)

            if num_flipped == 0:
                cost = 0
            else:
                cost = cost_sum / num_flipped

            label = f"{eff=:.2%}\n{cost=:.2f}\n{size=}\n"
            for action in actions_:
                label += f"{action}\n"

            graph.add_vertex(
                node.id,
                label=label,
            )
            for _child_name, child in node.children.items():
                add_nodes(child)

        add_nodes(self)

        def add_edges(node):
            for _child_name, child in node.children.items():
                graph.add_edge(
                    node.id, child.id, label=f"{node.split_feature} {child.data_val}"
                )
                add_edges(child)

        add_edges(self)

        return graph

    def display_igraph_jupyter(self, numeric_features=[]):
        """
        Displays the tree structure in a Jupyter notebook using matplotlib and igraph.

        Parameters:
        ----------
        numeric_features : list
            A list of numeric feature names used for processing node labels.
        """
        import igraph as ig
        import matplotlib.pyplot as plt

        g = self.to_igraph(numeric_features=numeric_features)
        fig, ax = plt.subplots()
        fig.set_figheight(10)
        fig.set_figwidth(40)
        vertex_labels = g.vs["label"]
        edge_labels = g.es["label"]
        ig.plot(
            g,
            target=ax,
            layout="reingold_tilford",
            vertex_size=55,
            # vertex_frame_width=10.0,
            # vertex_frame_color="white",
            vertex_label=vertex_labels,
            edge_label=edge_labels,
            vertex_label_size=8.0,
        )
        ax.invert_yaxis()
    