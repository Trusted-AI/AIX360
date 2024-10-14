class Node:
    def __init__(
        self,
        split_feature=None,
        actions=None,
        effectiveness=0,
        cost=0,
        size=0,
    ):
        self.split_feature = split_feature
        self.effectiveness = effectiveness
        self.cost = cost
        self.size = size
        self.children = {}
        self.actions = actions

    def add_child(self, subgroup, child_node):
        print()
        self.children[tuple(subgroup)] = child_node

    def return_leafs_actions(self):
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
        import igraph as ig
        import matplotlib.pyplot as plt

        g = self.to_igraph(numeric_features=numeric_features)
        fig, ax = plt.subplots()
        fig.set_figwidth(20)
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
    