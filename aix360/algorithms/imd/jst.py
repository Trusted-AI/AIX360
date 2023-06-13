import numpy as np
import pandas as pd

from aix360.algorithms.imd.rule import Rule


def leaf_to_rule(leaf):
    """
    converts a leaf node dictionary to a Rule object.
    Args:
        leaf:

    Returns:
    """
    return Rule(0, leaf['path'], leaf['val'])


def get_leaves(root: dict):
    """
    returns the leaf nodes (list of dicts) from a single surrogate tree.
    :param root:
    :return:
    """
    nodes = []
    def _recurse(root):
        is_split_node = 'cutoff' in root
        if is_split_node:
            _recurse(root['left'])
            _recurse(root['right'])
        else:
            nodes.append(root)

    _recurse(root)
    return nodes


class JointSurrogateTree:
    """
    Jointly trains 2 surrogate decision trees.
    Refines diff (or sim) regions by sample generation.
    """

    def __init__(self, max_depth, feature_names, alpha=0.0, split_criterion=1, refine=False):
        """
        Args:
            max_depth: maximum depth of the joint surrogate tree to be built
            feature_names: list of input feature names
            alpha: parameter to control degree of favouring common nodes vs. separate nodes
            split_criterion: which divergence criterion to use? (see paper for more details)
            refine: not used here
        """
        self.depth1 = 0
        self.depth2 = 0
        self.alpha = alpha
        self.max_depth = max_depth
        self.feature_names = feature_names  # considered common
        # output two decision trees
        self.tree1 = None
        self.tree2 = None
        if split_criterion == 1:
            # default is alpha=0, diverge if atleast one tree is pure
            self.continue_same_prefix = self.split_condition
        else:
            self.continue_same_prefix = self.split_condition2

        self.refine = refine  # unused
        self.generated_data = {}
        self.diffrules = []

    def H(self, y, mode='entropy'):
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        if y.shape[0] == 0:
            return 0.0
        classes = np.unique(y)

        p = (y == classes[:, None]).sum(axis=1)
        p = p / y.shape[0]

        ''' entropy range = [0, 1]
             gini   range = [0, 1/2]
             0 --> pure
             1 --> impure
        '''
        if mode == 'entropy':
            eps = 1e-8
            return -(p * np.log2(p + eps)).sum()
        else:
            return (p * (1 - p)).sum()

    def get_entropy_split(self, y_predict, y_real, mode='entropy'):
        """
        y_predict is a boolean array representing the split.
        """
        assert len(y_predict) == len(y_real), f"length differs in labels! {len(y_predict)} != {len(y_real)}"

        y_left = y_real[y_predict]
        y_right = y_real[~y_predict]

        n_l = y_left.shape[0]
        n_r = y_right.shape[0]

        n = n_l + n_r
        p_left = n_l / n
        p_right = n_r / n

        return p_left * self.H(y_left, mode) + p_right * self.H(y_right, mode)

    def fit1(self, x, y, path=None, depth=0):
        if path is None:
            path = []
        if len(y) == 0:
            return None
        elif self.all_same(y):
            return {'val': y[0], 'depth': depth, 'ispure': self.all_same(y), 'dist': np.bincount(y), 'path': path}
        elif depth >= self.max_depth:
            # compute majority class in y
            label = np.argmax(np.bincount(y))
            return {'val': label, 'depth': depth, 'ispure': self.all_same(y), 'dist': np.bincount(y), 'path': path}
        else:
            col, cutoff, entropy = self.find_best_split_of_all(x, y)
            y_left = y[x[:, col] < cutoff]
            y_right = y[x[:, col] >= cutoff]

            left_path = path + [(self.feature_names[col], '<', np.around(cutoff, 4))]
            right_path = path + [(self.feature_names[col], '>=', np.around(cutoff, 4))]

            par_node = {'col': self.feature_names[col], 'index_col': col, 'cutoff': cutoff,
                        'val': np.argmax(np.bincount(y)), 'depth': depth, 'dist': np.bincount(y),
                        # recursive calls
                        'left': self.fit1(x[x[:, col] < cutoff], y_left, left_path, depth + 1),
                        'right': self.fit1(x[x[:, col] >= cutoff], y_right, right_path, depth + 1)}

            return par_node

    def fit(self, x1, y1, x2, y2, path=None, depth=0):
        if path is None:
            path = []
        # length check
        if len(y1) == 0 and len(y2) == 0:
            return None, None
        if len(y1) == 0:
            return None, self.fit1(x2, y2, path, depth)
        elif len(y2) == 0:
            return self.fit1(x1, y1, path, depth), None

        # all same check
        if self.all_same(y1) and self.all_same(y2):
            return {'val': y1[0], 'depth': depth, 'ispure': True, 'dist': np.bincount(y1), 'path': path},\
                   {'val': y2[0], 'depth': depth, 'ispure': True, 'dist': np.bincount(y2), 'path': path}

        if self.all_same(y1):
            return {'val': y1[0], 'depth': depth, 'ispure': True, 'dist': np.bincount(y1), 'path': path},\
                   self.fit1(x2, y2, path, depth)
        if self.all_same(y2):
            return self.fit1(x1, y1, path, depth),\
                   {'val': y2[0], 'depth': depth, 'ispure': True, 'dist': np.bincount(y2), 'path': path}

        if depth >= self.max_depth:
            # find majority class labels
            label1 = np.argmax(np.bincount(y1))
            label2 = np.argmax(np.bincount(y2))
            return {'val': label1, 'depth': depth, 'ispure': self.all_same(y1), 'dist': np.bincount(y1), 'path': path},\
                   {'val': label2, 'depth': depth, 'ispure': self.all_same(y2), 'dist': np.bincount(y2), 'path': path}

        col1, cutoff1, entropy1 = self.find_best_split_of_all(x1, y1)
        col2, cutoff2, entropy2 = self.find_best_split_of_all(x2, y2)
        col, cutoff, entropy = self.find_best_split_of_all_double(x1, y1, x2, y2)

        # print(col1, cutoff1, entropy1, col2, cutoff2, entropy2, col, cutoff, entropy)

        if self.continue_same_prefix(col1, cutoff1, entropy1, col2, cutoff2, entropy2, col, cutoff, entropy):
            # print(f"cont. with e1={entropy1}, e2={entropy2}, e={entropy}")

            # initialize 2 par_nodes corresponding to two trees
            par_node1 = {'col': self.feature_names[col], 'index_col': col, 'cutoff': cutoff,
                         'val': np.argmax(np.bincount(y1)), 'depth': depth, 'dist': np.bincount(y1)}
            # print('common',par_node1)
            par_node2 = {'col': self.feature_names[col], 'index_col': col, 'cutoff': cutoff,
                         'val': np.argmax(np.bincount(y2)), 'depth': depth, 'dist': np.bincount(y2)}

            # splits for tree1
            y1_left = y1[x1[:, col] < cutoff]
            y1_right = y1[x1[:, col] >= cutoff]

            # splits for tree2
            y2_left = y2[x2[:, col] < cutoff]
            y2_right = y2[x2[:, col] >= cutoff]

            # create paths
            left_path = path + [(self.feature_names[col], '<', np.around(cutoff, 4))]
            right_path = path + [(self.feature_names[col], '>=', np.around(cutoff, 4))]

            par_node1['left'], par_node2['left'] = self.fit(x1[x1[:, col] < cutoff], y1_left, x2[x2[:, col] < cutoff],
                                                            y2_left, left_path, depth + 1)
            par_node1['right'], par_node2['right'] = self.fit(x1[x1[:, col] >= cutoff], y1_right,
                                                              x2[x2[:, col] >= cutoff], y2_right, right_path, depth + 1)

            self.depth1 += 1
            self.depth2 += 1

            self.tree1 = par_node1
            self.tree2 = par_node2
        else:
            print(f"diverge with e1={entropy1}, e2={entropy2}, e={entropy}")
            # for tree1, use col1, cutoff1 to partition the data
            y1_left = y1[x1[:, col1] < cutoff1]
            y1_right = y1[x1[:, col1] >= cutoff1]
            # create paths
            left_path1 = path + [(self.feature_names[col1], '<', np.around(cutoff1, 4))]
            right_path1 = path + [(self.feature_names[col1], '>=', np.around(cutoff1, 4))]
            par_node1 = {'col': self.feature_names[col1], 'index_col': col1, 'cutoff': cutoff1,
                         'val': np.argmax(np.bincount(y1)), 'depth': depth, 'dist': np.bincount(y1),
                         # recursive calls for tree1
                         'left': self.fit1(x1[x1[:, col1] < cutoff1], y1_left, left_path1, depth + 1),
                         'right': self.fit1(x1[x1[:, col1] >= cutoff1], y1_right, right_path1, depth + 1)}

            # print('1',par_node1)

            # for tree2, use col2, cutoff2 to partition the data
            y2_left = y2[x2[:, col2] < cutoff2]
            y2_right = y2[x2[:, col2] >= cutoff2]
            # create paths
            left_path2 = path + [(self.feature_names[col2], '<', np.around(cutoff2, 4))]
            right_path2 = path + [(self.feature_names[col2], '>=', np.around(cutoff2, 4))]
            par_node2 = {'col': self.feature_names[col2], 'index_col': col2, 'cutoff': cutoff2,
                         'val': np.argmax(np.bincount(y2)), 'depth': depth, 'dist': np.bincount(y2),
                         # recursive calls for tree2
                         'left': self.fit1(x2[x2[:, col2] < cutoff2], y2_left, left_path2, depth + 1),
                         'right': self.fit1(x2[x2[:, col2] >= cutoff2], y2_right, right_path2, depth + 1)}

            # print('2',par_node2)

            self.depth1 += 1
            self.depth2 += 1

        return par_node1, par_node2

    def all_same(self, y: np.ndarray):
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        return (y[0] == y).all()

    def find_best_split_of_all(self, x: np.ndarray, y: np.ndarray):
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        col = None
        min_entropy = 10
        cutoff = None

        for idx, c in enumerate(x.T):
            values = np.unique(c)
            split_points = (values[:-1] + values[1:]) / 2
            # print(idx, values[0:5], split_points[0:5])
            for value in split_points:
                y_predict = c < value
                # print(f"y_predict len = {y_predict.shape}, y shape={y.shape}")
                cur_entropy = self.get_entropy_split(y_predict, y)

                if cur_entropy == 0:
                    return idx, value, cur_entropy

                elif cur_entropy <= min_entropy:
                    min_entropy = cur_entropy
                    col = idx
                    cutoff = value

        return col, cutoff, min_entropy

    def find_best_split_of_all_double(self, x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray):

        if isinstance(x1, pd.DataFrame):
            x1 = x1.to_numpy()
        if isinstance(y1, pd.Series):
            y1 = y1.to_numpy()
        if isinstance(x2, pd.DataFrame):
            x2 = x2.to_numpy()
        if isinstance(y2, pd.Series):
            y2 = y2.to_numpy()

        col = None
        min_entropy = 10
        cutoff = None

        for idx, c in enumerate(x1.T):
            values = np.unique(c)
            split_points = (values[:-1] + values[1:]) / 2

            for value in split_points:
                y_predict = c < value
                cur_entropy1 = self.get_entropy_split(y_predict, y1)
                cur_entropy2 = self.get_entropy_split(y_predict, y2)
                cur_entropy = (cur_entropy1 + cur_entropy2) / 2

                # choose a split that minimizes entropy of sum of the two
                if cur_entropy == 0:
                    return idx, value, cur_entropy

                elif cur_entropy <= min_entropy:
                    min_entropy = cur_entropy
                    col = idx
                    cutoff = value

        return col, cutoff, min_entropy

    def split_condition(self, col1, cutoff1, entropy1, col2, cutoff2, entropy2, col, cutoff, entropy):
        if entropy1 <= 0.0:  # floating point/1e-8 error, -1e-8 is zero
            return False
        if entropy2 <= 0.0:
            return False
        return True

    def split_condition2(self, col1, cutoff1, entropy1, col2, cutoff2, entropy2, col, cutoff, entropy):
        if entropy1 < 0 or entropy2 < 0 or entropy < 0:  # floating point/1e-8 error
            return True

        e_avg = (entropy1 + entropy2) / 2
        alpha_e_joint = self.alpha * entropy

        return alpha_e_joint < e_avg

    def predict(self, x, tree):
        results = np.array([0] * len(x))
        for i, c in enumerate(x):
            # print(i,c)
            results[i] = self._get_prediction(c, tree)
        return results

    def accuracy(self, x, y_true, tree):
        from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
        y_pred = self.predict(x, tree)
        acc = accuracy_score(y_true, y_pred)
        return np.around(acc, 4) * 100

    def _get_prediction(self, row, tree):
        cur_layer = tree
        # print(cur_layer['col'],cur_layer['index_col'])
        while cur_layer and ('cutoff' in cur_layer):
            # print(cur_layer['col'],cur_layer['index_col'], cur_layer['cutoff'], row[cur_layer['index_col']])
            if row[cur_layer['index_col']] < cur_layer['cutoff']:
                if not cur_layer['left']:  # true if `cur_layer['left'] is None`
                    return cur_layer['val']
                cur_layer = cur_layer['left']
            else:
                if not cur_layer['right']:
                    return cur_layer['val']
                cur_layer = cur_layer['right']
        else:
            return cur_layer.get('val')

    def common_trunk(self, t1, t2):
        """
        Args:
            t1: surrogate tree 1
            t2: surrogate tree 2

        Returns:
            combine common nodes and split diverging trees under or nodes

            returns a single tree dictionary capturing both the common trunk
            and diverging branches. The diverging trees are stored in `tree1` and `tree2`
            keys of t, for each diverging juncture.
        """

        """ gets the common prefix from two trees t1 and t2 and
        returns a single tree dictionary capturing both the common trunk
        and diverging branches. The diverging trees are stored in `tree1` and `tree2`
        keys of t, for each diverging juncture.
        """
        t = {}

        keys1 = t1.keys()
        keys2 = t2.keys()

        same_keys = keys1 == keys2
        is_split_node = 'cutoff' in t1

        if same_keys and is_split_node:
            same_split_cond = (t1['col'] == t2['col']) and (t1['cutoff'] == t2['cutoff'])
        else:
            same_split_cond = False

        if same_keys and is_split_node and same_split_cond:
            # same split node
            for key in t1:
                if key not in ['left', 'right']:
                    t[key] = t1[key]

            t['left'] = self.common_trunk(t1['left'], t2['left'])
            t['right'] = self.common_trunk(t1['right'], t2['right'])

        else:
            # diverging point: leaf nodes or different split nodes or one leaf and another split
            t['tree1'] = t1
            t['tree2'] = t2

        return t

    def get_diffrules_from_jst(self, root: dict):
        """
        takes the common joint surrogate tree representation
        :param root:
        :return:
        """
        diffrules = []

        def _recurse(root: dict):

            is_diverging = 'tree1' in root

            if is_diverging:
                # print('at one juncture...')
                tree1 = root['tree1']
                tree2 = root['tree2']

                diffrules.extend(self._get_diffrules(tree1, tree2))
                # print(f'tot #diffrules here = {len(diffrules)}')

            else:
                is_split_node = 'cutoff' in root
                if is_split_node:
                    _recurse(root['left'])
                    _recurse(root['right'])
                else:
                    # leaf node, do nothing
                    pass

        _recurse(root)
        return diffrules

    def _get_diffrules(self, root1: dict, root2: dict):
        leaves1 = get_leaves(root1)
        leaves2 = get_leaves(root2)

        # print(f"nl1 = {len(leaves1)}, nl2 = {len(leaves2)}")

        diffrules = []

        for leaf1 in leaves1:
            for leaf2 in leaves2:

                if leaf1['val'] != leaf2['val']:
                    rule1 = leaf_to_rule(leaf1)
                    rule2 = leaf_to_rule(leaf2)
                    intsec = rule1.intersection(rule2)
                    # print(rule1, rule2, rule1.as_dict())
                    # print('intsec', intsec)

                    if len(intsec.predicates) > 0:
                        diffrules.append(intsec)

        return diffrules

