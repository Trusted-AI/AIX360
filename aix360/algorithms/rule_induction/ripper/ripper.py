import logging
import logging.config
from collections import OrderedDict

import numpy as np
from aix360.algorithms.dise import DISExplainer
from aix360.algorithms.rule_induction.ripper.base import _encoding_for_parallel, init_encoder, encode_nominal, \
    _split_instances
from aix360.algorithms.rule_induction.ripper.binding import append_literal, _bind_literal, _bind_rule, \
    unbound_rule_list_index, _unbound, _unbound_rule_list, _rule_list_predict, LE, EQ, _find_best_literal, Literal, \
    _filter_contradicted_instances
from aix360.algorithms.rule_induction.ripper.mdl import _mdl
from aix360.algorithms.rule_induction.ripper.pruning import _pruning_irep, _minimize_rule_list, _pruning_optimization
from aix360.algorithms.rule_induction.trxf.core.conjunction import Conjunction
from aix360.algorithms.rule_induction.trxf.core.dnf_ruleset import DnfRuleSet
from aix360.algorithms.rule_induction.trxf.core.feature import Feature
from aix360.algorithms.rule_induction.trxf.core.predicate import Predicate, Relation
from pandas import DataFrame, Series

print('Importing dev version v0.982 of RIPPER')


class RipperExplainer(DISExplainer):
    """
    RIPPER (Repeated Incremental Pruning to Produce Error Reduction) is a heuristic rule induction algorithm
    based on separate-and-conquer. The explainer outputs a rule set in Disjunctive Normal Form (DNF) for a single
    target concept.

    References:
        .. [#ML95] `William W Cohen, "Fast Effective Rule Induction"
            Machine Learning: Proceedings of the Twelfth International Conference, 1995.
            <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.107.2612&rep=rep1&type=pdf>
    """

    def __init__(
            self,
            d: int = 64,
            k: int = 2,
            pruning_threshold: int = 20,
            random_state: int = 0,
    ):
        """
        Args:
            d (int): The number of bits that a new rule need to gain. Defaults to 64.
            k (int): The number of iterations for the optimization loop. Defaults to 2.
            pruning_threshold (int): The minimum number of instances for splitting. Defaults to 20.
            random_state (int): The random seed for the splitting function. Defaults to 0.
        """

        super().__init__()
        self.d = d
        self.k = k
        self.pruning_threshold = pruning_threshold
        self.random_state = random_state
        self._target_label = None
        self.default_label = None

    @property
    def target_label(self):
        """
        The latest positive value RIPPER has been fitted for.
        """
        return self._target_label

    def set_params(self, **kwargs):
        self.d = kwargs.get('d', 64)
        self.k = kwargs.get('k', 2)
        self.pruning_threshold = kwargs.get('pruning_threshold', 20)
        self.random_state = kwargs.get('random_state', 0)

    def fit(self, train: DataFrame, y: Series, target_label=None):
        """
        The fit function for RIPPER algorithm. Its implementation is limited to DataFrame and Series because the RIPPER
        algorithm needs the information of feature name and have to support nominal data type. Only `float` dtypes are
        considered numerical features. All others (including int) are treated as nominal.

        If target_label is specified, binary classification is assumed and asserted, and training uses target_label as selection\
        of positive examples.

        The induction of rules is deterministic by default as all random choices are initialized with self.random_state,
        which is 0 by default.

        Args:
            train (pd.DataFrame): The features of the training set
            y (pd.Series): The labels of the training set
            target_label (Any): The target label to learn for binary classification, among the unique values of y. \
            If not provided, Ripper will induce a native ordered ruleset with multiple labels/conclusions.

        Returns:
            self
        """
        logger = logging.getLogger(__name__)

        self._target_label = target_label

        self._labels = None
        self._rule_map = None

        self._nominal_column_encoders = None
        self._column_name_list = None
        self._column_name_index_map = None
        self._condition_mat = None

        self._gain_n = 0

        # Check that X and y have correct shape
        assert train.shape[0] == y.shape[0], 'X, y should have same length'

        self._nominal_column_encoders = init_encoder(train)
        encode_nominal(self._nominal_column_encoders, train)
        self._column_name_list = train.columns.values

        self._column_name_index_map = {name: index for index, name in enumerate(self._column_name_list)}

        unique_labels, counts = np.unique(y, return_counts=True)
        label_count_dict = dict(zip(unique_labels, counts))

        assert len(unique_labels) > 1, 'label has only one value'

        # Check whenever target_label is given that label has only two values and target_label is one of them
        assert (target_label is None) or (len(unique_labels) == 2), 'Positive label value given but label not binary.'
        assert (target_label is None) or (
                target_label in unique_labels), 'Positive label value given is not a label value.'

        self._labels = sorted(unique_labels, key=lambda e: label_count_dict[e],
                              reverse=False)  # returns list from ndarray;
        # the label values are sorted according to their occurrence; least frequent value first
        # Set reverse=True for reversing that order
        if (target_label is not None) and (self._labels[0] != target_label):
            # reverse value order for binary classification when target_label is not the least frequent
            self._labels.reverse()
        self.default_label = self._labels[-1]  # the default value is the last label in the sorting order
        logger.debug('Label value ordering: ' + str(self._labels))

        self._condition_mat = _encoding_for_parallel(train, self._column_name_index_map)
        self._gain_n = self._condition_mat[0].shape[0]

        train = train.to_numpy(dtype=np.dtype(float))
        y = y.values

        x_irep_list = train
        y_irep = y

        self._rule_map = OrderedDict()

        for i in range(len(self._labels) - 1):
            pos = self._labels[i]
            neg = self._labels[i + 1:]

            indexer_pos = y_irep == pos
            indexer_neg = np.isin(y_irep, neg)

            train_pos = x_irep_list[indexer_pos]
            train_neg = x_irep_list[indexer_neg]

            irep_res = self._irep_plus_outer_loop(
                train_pos,
                train_neg,
                d=self.d,
                ratio=2 / 3,
                pruning_threshold=self.pruning_threshold
            )
            if len(irep_res) > 0:
                self._rule_map[pos] = irep_res
                bool_vec_index = unbound_rule_list_index(x_irep_list, self._rule_map[pos])
                x_irep_list = x_irep_list[bool_vec_index]
                y_irep = y_irep[bool_vec_index]

        logger.debug('# begin optimization')

        for _ in range(self.k):
            x_optimize = train
            y_optimise = y
            for i in range(len(self._labels) - 1):
                if not self._rule_map.get(self._labels[i]):
                    continue
                pos = self._labels[i]
                neg = self._labels[i + 1:]

                indexer_pos = y_optimise == pos
                indexer_neg = np.isin(y_optimise, neg)

                train_pos = x_optimize[indexer_pos]
                train_neg = x_optimize[indexer_neg]

                self._rule_map[pos] = self._optimize(
                    train_pos,
                    train_neg,
                    ratio=2 / 3,
                    rules_input=self._rule_map[pos]
                )

                bool_vec_index = unbound_rule_list_index(x_optimize, self._rule_map[pos])
                x_optimize = x_optimize[bool_vec_index]
                y_optimise = y_optimise[bool_vec_index]

        return self

    def predict(self, X: DataFrame) -> np.ndarray:
        """
        The predict function for RIPPER algorithm. Its implementation is limited to DataFrame and Series because the
        RIPPER algorithm needs the information of feature name and have to support nominal data type

        Args:
            X (pd.DataFrame): DataFrame of features

        Returns:
            np.array: predicted labels
        """
        test = X.copy()
        encode_nominal(self._nominal_column_encoders, test)
        result_vec = [None for _ in range(len(test))]

        input_arr = test.to_numpy(dtype=np.dtype(float))

        result = {label: _rule_list_predict(input_arr, self._rule_map[label]) for label in self._rule_map.keys()}

        for label in self._rule_map.keys():
            for i in range(len(result_vec)):
                if result_vec[i] is None and result[label][i]:
                    result_vec[i] = label

        return np.array([res if res is not None else self.default_label for res in result_vec])

    def explain(self):
        """
        Export rule set to technical interchange format trxf from internal representation
        for the positive value (i.e. label value) it has been fitted for.

        When the internal rule set is empty an empty dnf rule set with the internal pos value
        is returned.

        Returns:
            trxf.DnfRuleSet
        """
        assert (self.target_label is not None), 'Not fitted or not fitted for a specific pos value. Use export_rules ' \
                                                'in the latter case. '

        if len(self._rule_map.items()) == 0:
            return DnfRuleSet([], self.target_label)
        for label, rules in self._rule_map.items():
            if label == self.target_label:
                return self._rules_to_trxf_dnf_ruleset(rules, label)
        raise Exception('No rules found for label: ' + str(self.target_label))

    def explain_multiclass(self):
        """
        Export rules to technical interchange format trxf from internal representation
        Returns a list of rule sets.

        Returns:
            list(trxf.DnfRuleSet): -- Ordered list of rulesets
        """
        res = list()
        if len(self._rule_map.items()) == 0:
            return DnfRuleSet([], self.target_label)
        for label, rules in self._rule_map.items():
            dnf_ruleset = self._rules_to_trxf_dnf_ruleset(rules, label)
            res.append(dnf_ruleset)
        default_rule = DnfRuleSet([], self.default_label)
        res.append(default_rule)
        return res

    def _irep_plus_outer_loop(
            self,
            pos: np.ndarray,
            neg: np.ndarray,
            d: int,
            ratio: float = 2 / 3,
            pruning_threshold: int = 20
    ):
        """
        The learning phase of RIPPER.

        Args:
            pos (np.ndarray): Positive instances
            neg (np.ndarray): Negative instances
            d (int): The number to bit that a new rule need to gain
            ratio (float): The percentage of pruning data
            pruning_threshold (int): The minimum number of instances for splitting

        Returns:
            list: Rule list learned using IREP algorithm
        """
        neg = _filter_contradicted_instances(pos, neg)
        n = self._gain_n

        rules = []

        # TODO verify if the intent was to make a copy of the value instead of the reference
        pos_original = pos
        neg_original = neg

        dl_min = _mdl(pos, neg, rules, n)

        while len(pos) > 0:

            if len(pos) > pruning_threshold:
                pos_grow, pos_prune, neg_grow, neg_prune = _split_instances(pos, neg, ratio, self.random_state)
                rule = self._grow_rule(pos_grow, neg_grow)
                rule = _pruning_irep(pos_prune, neg_prune, rule)
            else:
                rule = self._grow_rule(pos, neg)

            rules.append(rule)
            dl_new = _mdl(pos, neg, rules, n)

            if dl_new > dl_min + d:
                return _minimize_rule_list(pos_original, neg_original, rules, n)
            else:
                if dl_new < dl_min:
                    dl_min = dl_new
                pos = _unbound(pos, rule)
                neg = _unbound(neg, rule)

        return rules

    def _replacement(
            self,
            pos: np.ndarray,
            neg: np.ndarray,
            rules,
            index: int,
            ratio: float = 2 / 3
    ):
        """
        The replacement step for RIPPER optimization

        Parameters
        ----------
            pos: np.ndarray
                Positive instances
            neg: np.ndarray
                Negative instances
            rules : list
                Rules that needs to be optimized
            index : int
                Index of selected rule
            ratio : float
                The percentage of pruning data

        Returns
        -------
        list
            A replacement rule
        """
        rest = rules[:index] + rules[index + 1:]

        new_pos = _unbound_rule_list(pos, rest)

        if len(new_pos) > 2:

            pos_grow, pos_prune, neg_grow, neg_prune = _split_instances(new_pos, neg, ratio, self.random_state)

            new_rule = self._grow_rule(pos=pos_grow, neg=neg_grow)

            pruned_rule = _pruning_optimization(
                pos_prune=pos_prune,
                neg_prune=neg_prune,
                rule=new_rule,
                rules=rules,
                index=index
            )
            return pruned_rule
        else:
            return rules[index]

    def _revision(
            self,
            pos: np.ndarray,
            neg: np.ndarray,
            rules,
            index: int,
            ratio: float = 2 / 3
    ):
        """
        The revision step for RIPPER optimization

        Parameters
        ----------
            pos: np.ndarray
                Positive instances
            neg: np.ndarray
                Negative instances
            rules : list
                Rules that needs to be optimized
            index : int
                Index of selected rule
            ratio : float
                The percentage of pruning data

        Returns
        -------
        list
            A revision rule
        """
        rule = rules[index]

        new_pos = _unbound_rule_list(pos, rules[:index] + rules[index + 1:])

        if new_pos.shape[0] > 2:

            pos_grow, pos_prune, neg_grow, neg_prune = _split_instances(new_pos, neg, ratio, self.random_state)
            new_rule = self._grow_rule(pos_grow, neg_grow, predefined_rule=rule)
            pruned_rule = _pruning_optimization(
                pos_prune=pos_prune,
                neg_prune=neg_prune,
                rule=new_rule,
                rules=rules,
                index=index
            )
            return pruned_rule
        else:
            return rule

    def _optimize(
            self,
            pos: np.ndarray,
            neg: np.ndarray,
            rules_input,
            ratio: float = 2 / 3
    ):
        """
        The optimization step for RIPPER

        Parameters
        ----------
            pos: np.ndarray
                Positive instances
            neg: np.ndarray
                Negative instances
            rules_input : list
                Rules that needs to be optimized
            ratio : float
                The percentage of pruning data

        Returns
        -------
        list
            Optimized rule list
        """
        rules = rules_input.copy()

        i = 0
        while i < len(rules):

            # ====================== origin ============================================================================
            dl_origin = _mdl(pos, neg, rules, self._gain_n)

            # ====================== replacement =======================================================================
            replacement_rule = self._replacement(pos=pos, neg=neg, rules=rules, ratio=ratio, index=i)
            if len(replacement_rule) != 0:
                dl_replacement = _mdl(pos, neg, rules[:i] + [replacement_rule] + rules[i + 1:], self._gain_n)
            else:
                dl_replacement = _mdl(pos, neg, rules[:i] + rules[i + 1:], self._gain_n)

            # ====================== revision ==========================================================================
            revision_rule = self._revision(pos=pos, neg=neg, rules=rules, ratio=ratio, index=i)

            dl_revision = _mdl(pos, neg, rules[:i] + [revision_rule] + rules[i + 1:], self._gain_n)

            # ====================== result ============================================================================
            if dl_origin <= dl_replacement and dl_origin <= dl_revision:
                # if no improvement is made, then the original rule is chosen
                pass
            # secondly, replacement rule is tested
            elif dl_replacement <= dl_origin and dl_replacement <= dl_revision:
                if len(replacement_rule) != 0:
                    rules[i] = replacement_rule
                else:
                    del rules[i]
                    continue
            # finally, if the origin rule and the replacement rule are not satisfied, the revision rule is chosen
            else:
                rules[i] = revision_rule
            i += 1
        return rules

    def _rule_to_trxf_conjunction(self, rule):
        """
        Transform one rule to a trxf conjunction given its internal presentation

        Parameters
        ----------
        rule : list
            Input rule

        Returns
        -------
        trxf conjunction
            String representation of that rule
        """
        # TODO code duplication with decode_rule should be refactored -> deprecate decode_rule as it should go
        #  through trxf

        conjunction = Conjunction([])

        for condition in rule:
            name = self._column_name_list[condition.name]
            feature = Feature(name)
            if condition.op == EQ:
                relation = Relation.EQ
                value = self._nominal_column_encoders[name].classes_[condition.nom_val]
            elif condition.op == LE:
                relation = Relation.LE
                value = condition.num_val
            else:
                relation = Relation.GE
                value = condition.num_val

            predicate = Predicate(feature, relation, value)
            conjunction.add_predicate(predicate)

        return conjunction

    def _rules_to_trxf_dnf_ruleset(self, rules, label):
        """
        Transform rules to trxf dnf_ruleset given their internal presentation and their label

        Parameters
        ----------
        rules : list
            Rules for one target
        label : str
            The label of rules

        Returns
        -------
            DnfRuleSet
        """
        conjunctions = list()
        for rule in rules:
            conjunction = self._rule_to_trxf_conjunction(rule)
            conjunctions.append(conjunction)
        dnf_ruleset = DnfRuleSet(conjunctions, label)
        return dnf_ruleset

    def _grow_rule(self, pos, neg, predefined_rule=None):

        """
        The grow function for IREP*

        Parameters
        ----------
        pos : np.ndarray
            Positive instances
        neg : np.ndarray
            Negative instances
        predefined_rule: const int64_t[::1]
            Existing rules

        Returns
        -------
        list
            Learned rule
        """
        _validate_grow_rule_input(pos)
        neg = _filter_contradicted_instances(pos, neg)

        if predefined_rule is not None:
            pos = _bind_rule(pos, predefined_rule)
            neg = _bind_rule(neg, predefined_rule)

            # the length of pos must be always positive
            if len(pos) == 0 or len(neg) == 0:
                return predefined_rule

            learned_rule = list(predefined_rule)
        else:
            learned_rule = list()

        while len(neg) > 0:
            pos = np.asfortranarray(pos)
            neg = np.asfortranarray(neg)

            best_literal_index = _find_best_literal(pos,
                                                    neg,
                                                    self._condition_mat[0],
                                                    self._condition_mat[1],
                                                    self._condition_mat[2],
                                                    self._condition_mat[3]
                                                    )
            literal = Literal(self._condition_mat[0][best_literal_index],
                              self._condition_mat[1][best_literal_index],
                              self._condition_mat[2][best_literal_index],
                              self._condition_mat[3][best_literal_index]
                              )

            append_literal(learned_rule, literal)
            pos = _bind_literal(pos, literal.name, literal.op, literal.num_val, literal.nom_val)
            neg = _bind_literal(neg, literal.name, literal.op, literal.num_val, literal.nom_val)
        return learned_rule


def _validate_grow_rule_input(pos):
    if len(pos) == 0:
        raise AssertionError('pos must not be empty')
