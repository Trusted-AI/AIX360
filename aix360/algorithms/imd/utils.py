import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict

from aix360.algorithms.imd.rule import Rule


def decision_tree_to_rules(dt, feature_names, return_type="dict"):
    """
    utility to get rules from a scikit-learn decision tree model.
    Args:
        dt: scikit-learn `DecisionTreeClassifier` fitted model
        feature_names: list
        return_type: list/dictionary of rules

    Returns:
        list/dictionary of rules based on `return_type`
    """

    n_nodes = dt.tree_.node_count
    children_left = dt.tree_.children_left
    children_right = dt.tree_.children_right
    feature = dt.tree_.feature
    threshold = dt.tree_.threshold
    value = dt.tree_.value

    rules = dict()

    def dfs(node_id, rule):
        is_split_node = children_left[node_id] != children_right[node_id]
        if is_split_node:
            f = feature[node_id]
            t = threshold[node_id]
            name = feature_names[f]
            dfs(children_left[node_id], rule + [(name, '<=', t)])
            dfs(children_right[node_id], rule + [(name, '>', t)])
        else:
            class_distribution = value[node_id]
            label = np.argmax(value[node_id][0])  # take majority class' label
            rule = Rule(node_id, rule, label)
            rules[node_id] = rule
            # rules.append(rule)

    dfs(0, rule=[])
    if return_type == "dict":
        return rules
    return [item for (key, item) in rules.items()]


def _de_dummy_code_df(df, sep="=", set_category=False):
    """
    taken verbatim from aif360
    @param df: one hot encoded dataframe with column names like "race=White" etc.
    @param sep: separator of one hot encoded column names
    @param set_category:
    @return: de-dummy-fied dataframe. Categorical features' values will be strings.
    """
    feature_names_dum_d, feature_names_nodum = _parse_feature_names(df.columns)
    df_new = pd.DataFrame(index=df.index,
                          columns=feature_names_nodum + list(feature_names_dum_d.keys()))

    for fname in feature_names_nodum:
        df_new[fname] = df[fname].values.copy()

    for fname, vl in feature_names_dum_d.items():
        for v in vl:
            df_new.loc[df[fname + sep + str(v)] == 1, fname] = str(v)

    if set_category:
        for fname in feature_names_dum_d.keys():
            df_new[fname] = df_new[fname].astype('category')

    return df_new


def _parse_feature_names(feature_names, sep="="):
    """
    taken verbatim from aif360
    @param feature_names: list of column names.
    @param sep: separator of one hot encoded column names eg. =
    @return: returns the categorical features (defaultdict) and numerical features (list).
    """
    feature_names_dum_d = defaultdict(list)
    feature_names_nodum = list()
    for fname in feature_names:
        if sep in fname:
            fname_dum, v = fname.split(sep, 1)
            feature_names_dum_d[fname_dum].append(v)
        else:
            feature_names_nodum.append(fname)

    return feature_names_dum_d, feature_names_nodum


# all numerical dataset, multiclass
def load_iris_dataset():
    from sklearn.datasets import load_iris
    data = load_iris(as_frame=True)
    return data["data"], data["target"]


# all numerical dataset
def load_bc_dataset():
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer(as_frame=True)
    return data["data"], data["target"]


def common_trunk_tree_to_digraph(root: dict):
    """
    returns a networkx digraph from the common trunk tree dictionary (jst).
    Args:
        root: joint surrogate tree structure (a dictionary)

    Returns:
        T: a networkx digraph representation of the same jst
    """

    T = nx.DiGraph()
    color1 = "lightpink"    # node color for tree1
    color2 = "lightsalmon"  # node color for tree2

    leaf_colors = ["antiquewhite", "lightcyan", "grey70", "antiquewhite3", "aquamarine", "floralwhite"]  # assume max 6 classes

    def _recurse(root, parentid, direction, style="", fillcolor="lightgrey"):
        vnum = T.number_of_nodes()

        is_diverging = 'tree1' in root

        if is_diverging:
            # separate trees
            T.add_node(vnum, shape="point")

            _recurse(root["tree1"], vnum, '1', style="filled", fillcolor=color1)
            _recurse(root["tree2"], vnum, '2', style="filled", fillcolor=color2)

        else:
            # common part
            if 'cutoff' in root:
                # split node
                label = '{}({}) < {}'.format(root['col'], root['index_col'], np.around(root['cutoff'], 4))
                # print(label)
                T.add_node(vnum, label=label, style=style, fillcolor=fillcolor)

                if 'left' in root:
                    _recurse(root['left'], vnum, 'L', style, fillcolor)
                    _recurse(root['right'], vnum, 'R', style, fillcolor)

            else:
                # leaf node
                label = 'label={}\n#samples={}'.format(root['val'], root['dist'])
                T.add_node(vnum, label=label, pred=root['val'], shape="box", style='filled', fillcolor=leaf_colors[root['val']])

        if parentid is not None:
            if direction in ['1', '2']:
                # this is where the trees diverge
                T.add_edge(parentid, vnum, label=direction, style='dashed')
            else:
                # regular left-right edge
                T.add_edge(parentid, vnum, label=direction)

    _recurse(root, None, None)
    return T


or_nodes = 0
def simpler_common_trunk_tree_to_digraph(root: dict):
    """
    visualisation utility to prepare a networkx graph and draw it using graphviz.
    returns a networkx digraph from the common trunk tree dictionary (jst).
    Args:
        root: joint surrogate tree structure (a dictionary)

    Returns:
        T: a networkx digraph representation of the same jst
    """
    T = nx.DiGraph()
    color1 = "lightpink"  # node color for tree1
    color2 = "lightsalmon"  # node color for tree2

    leaf_colors = ["antiquewhite", "lightcyan", "grey70", "antiquewhite3", "aquamarine",
                   "floralwhite"]  # assume max 6 classes

    def _recurse(root, parentid, direction, style="", fillcolor="lightgrey"):
        vnum = T.number_of_nodes()

        is_diverging = 'tree1' in root

        if is_diverging:
            # separate trees
            global or_nodes
            # print(or_nodes)
            label = f"Vo[{or_nodes}]"
            T.add_node(vnum, shape="circle", label=f"vo{or_nodes}", style="dotted")
            or_nodes += 1

            _recurse(root["tree1"], vnum, '1', style="filled", fillcolor=color1)
            _recurse(root["tree2"], vnum, '2', style="filled", fillcolor=color2)

        else:
            # common part
            if 'cutoff' in root:
                # split node
                # label = 'X[{}] < {}'.format(root['index_col'], np.around(root['cutoff'], 2))
                label = '{}({}) < {}'.format(root['col'], root['index_col'], np.around(root['cutoff'], 2))
                # print(label)
                T.add_node(vnum, label=label, style=style, fillcolor=fillcolor)

                if 'left' in root:
                    _recurse(root['left'], vnum, 'T', style, fillcolor)
                    _recurse(root['right'], vnum, 'F', style, fillcolor)

            else:
                # leaf node
                label = 'label={}\n#samples={}'.format(root['val'], root['dist'])
                if root['ispure']:
                    s = "pure"
                else:
                    s = "impure"
                # label = '{}\n{}'.format(root['val'], s)
                label = '{}'.format(s)
                T.add_node(vnum, label=label, shape="box", style='filled', fillcolor=leaf_colors[root['val']])

        if parentid is not None:
            if direction in ['1', '2']:
                # this is where the trees diverge
                T.add_edge(parentid, vnum, label=direction, style='dashed')
            else:
                # regular left-right edge
                T.add_edge(parentid, vnum, label=direction)

    _recurse(root, None, None)
    # re-assign global or node counting variable to zero before returning
    global or_nodes
    or_nodes = 0
    return T


def graph_to_jpg(T, path='abcd.jpg'):
    """
    save the networkx digraph form of JST to an image.
    Args:
        T: networkx digraph object of the jst as returned from `simpler_common_trunk_tree_to_digraph` function
        path: path to save the jpg file.

    Returns:
        path of the saved image
    """
    from networkx.drawing.nx_agraph import to_agraph
    A = to_agraph(T)
    A.layout('dot')
    A.draw(path, format="jpg")
    return path


def visualize_jst(joint_surrogate_tree: dict, path='abcd.jpg'):
    T = simpler_common_trunk_tree_to_digraph(joint_surrogate_tree)
    return graph_to_jpg(T, path)
