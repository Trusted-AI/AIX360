import numpy as np
import numba as nb


LE = 0
GE = 1
EQ = 2


# # jitclass is not optimized for interpreter yet
# @nb.jitclass(
#     [
#         ('name', nb.int64),
#         ('op', nb.uint8),
#         ('num_val', nb.double),
#         ('nom_val', nb.int64)
#     ]
# )
class Literal(object):
    """
    A literal is an atomic formula (atom). In rule induction, the negation of an atom needs
    to be explicitly added. For example, ‘credit score >= 200’ and ‘credit score < 200’ (to solve
    some issues related to float number, the complement of >= is sometimes written as <=) need to
    be added at the same time

    name : int64_t
        Feature name for literal (encoded)
    op : uint8_t
        Operator for literal (encoded)
    num_val : double
        Numerical feature value for literal (encoded)
    nom_val : int64_t
        Nominal feature value for literal
    """

    def __init__(
            self,
            name,
            op,
            num_val,
            nom_val
    ):
        self.name = name
        self.op = op
        self.num_val = num_val
        self.nom_val = nom_val

    def __repr__(self):
        return '_'.join([
               str(self.name),
               str(self.op),
               str(self.num_val),
               str(self.nom_val)])

    def __str__(self):
        return self.__repr__()


def _filter_contradicted_instances(pos, neg):
    """
    Delete common instances in pos and neg

    Parameters
    ----------
    pos: const double[::1, :]
    neg: const double[::1, :]

    Returns
    -------
    const double[::1, :]
        New neg
    """
    feature_shape = np.shape(neg)[1]
    pos_set = set(map(tuple, pos))
    neg_set = set(map(tuple, neg))
    filtered_neg_set = neg_set.difference(pos_set)
    return neg if len(filtered_neg_set) == len(neg_set) else np.array(list(map(list, filtered_neg_set))).reshape(-1, feature_shape)


@nb.njit(
    'double(int64, int64, int64, int64)',
    nogil=True,
    cache=True,
    error_model='numpy'
)
def _gain(
        p0,
        p1,
        n0,
        n1
):
    """
    Calculation of p-foil gain for IREP*

    Parameters
    ----------
    p0 : int64
        Total positive instances
    p1 : int64
        Positive instances in the next iteration
    n0 : int64
        Total negative instances
    n1 : int64
        Negative instances in the next iteration

    Returns
    -------
    double
        The p-foil gian
    """
    if p1 == 0:
        return 0
    else:
        return p1 * (np.log2(p1 / (p1 + float(n1))) - np.log2(p0 / (p0 + float(n0))))


@nb.njit(
    [
        'int64(double[::1, :], int64, uint8, double, int64)',
        'int64(double[:, ::1], int64, uint8, double, int64)'
    ],
    nogil=True,
    cache=True,
    error_model='numpy'
)
def _count_bound_literal(
        data,
        name,
        op,
        num_val,
        nom_val,
):
    """
    Count the number of instances that are covered by the literal

    Parameters
    ----------
    data : const double[::1, :]
        Input instances
    name : int64_t
        Feature name for literal
    op : uint8_t
        Operator for literal
    num_val : double
        Numerical feature value for literal
    nom_val : int64_t
        Nominal feature value for literal

    Returns
    -------
    uint64_t
        The number of instances which are covered by the literal
    """
    result_count = 0
    if op == LE:
        for i in range(len(data)):
            result_count += data[i, name] <= num_val
    elif op == GE:
        for i in range(len(data)):
            result_count += data[i, name] >= num_val
    elif op == EQ:
        for i in range(len(data)):
            result_count += int(data[i, name]) == nom_val
    else:
        raise RuntimeError('Unknown operator')
    return result_count


@nb.njit(
    [
        'int64(double[::1, :], double[::1, :], int64[::1], uint8[::1], double[::1], int64[::1])',
        'int64(double[::1, :], double[:, ::1], int64[::1], uint8[::1], double[::1], int64[::1])',
        'int64(double[:, ::1], double[::1, :], int64[::1], uint8[::1], double[::1], int64[::1])',
        'int64(double[:, ::1], double[:, ::1], int64[::1], uint8[::1], double[::1], int64[::1])',
    ],
    nogil=True,
    parallel=True,
    error_model='numpy'
)
def _find_best_literal(
        pos,
        neg,
        name_vec,
        op_vec,
        num_val_vec,
        nom_val_vec
):
    """
    Find the best literal for OpenMP based IREP*

    Parameters
    ----------
    pos : const double[::1, :]
        Positive instances
    neg : const double[::1, :]
        Negative instances
    name_vec : const int64_t[::1]
        Feature names for literals
    op_vec : const uint8_t[::1]
        Operators for literals
    num_val_vec : const double[::1]
        Numerical feature values for literals
    nom_val_vec : const int64_t[::1]
        Nominal feature values for literals

    Returns
    -------
    uint64_t
        The index of the best literal
    """
    p = len(pos)
    n = len(neg)
    gain_list = np.empty(len(name_vec), dtype=np.float64)

    for i in nb.prange(len(name_vec)):

        name = name_vec[i]
        op = op_vec[i]
        num_val = num_val_vec[i]
        nom_val = nom_val_vec[i]

        gain_list[i] = _gain(
            p,
            _count_bound_literal(pos, name, op, num_val, nom_val),
            n,
            _count_bound_literal(neg, name, op, num_val, nom_val)
        )

    return np.argmax(gain_list)


@nb.njit(
    [
        'boolean[::1](double[::1, :], int64, uint8, double, int64)',
        'boolean[::1](double[:, ::1], int64, uint8, double, int64)',
     ],
    nogil=True,
    cache=True,
    error_model='numpy'
)
def bind_literal_index(
        data,
        name,
        op,
        num_val,
        nom_val,
):
    """
    Find instances that are covered by the literal

    Parameters
    ----------
    data : const double[::1, :]
        Input instances
    name : int64_t
        Feature name for literal
    op : uint8_t
        Operator for literal
    num_val : double
        Numerical feature value for literal
    nom_val : int64_t
        Nominal feature value for literal

    Returns
    -------
    uint64_t
        Instances which are covered by the literal
    """
    result = np.empty(len(data), dtype=np.bool_)
    if op == LE:
        for i in range(len(data)):
            result[i] = data[i, name] <= num_val
    elif op == GE:
        for i in range(len(data)):
            result[i] = data[i, name] >= num_val
    elif op == EQ:
        for i in range(len(data)):
            result[i] = int(data[i, name]) == nom_val
    else:
        raise RuntimeError('Unknown operator')
    return result


@nb.njit(
    [
        'double[:,::1](double[::1, :], int64, uint8, double, int64)',
        'double[:,::1](double[:, ::1], int64, uint8, double, int64)'
     ],
    nogil=True,
    cache=True,
    error_model='numpy'
)
def _bind_literal(
        data,
        name,
        op,
        num_val,
        nom_val,
):
    """
    Find instances that are covered by the literal

    Parameters
    ----------
    data : const double[::1, :]
        Input instances
    name : int64_t
        Feature name for literal
    op : uint8_t
        Operator for literal
    num_val : double
        Numerical feature value for literal
    nom_val : int64_t
        Nominal feature value for literal

    Returns
    -------
    ndarray
        Instances that are covered by the literal
    """
    return data[bind_literal_index(data, name, op, num_val, nom_val)]


def _bind_rule_index(
        data,
        rule
):
    """
    Find instances that are covered by the rule

    Parameters
    ----------
    data : const double[::1, :]
        Input instances
    rule : list
        Input rule

    Returns
    -------
    ndarray
        Boolean index array for instances that are covered by the rule
    """
    index = np.full(len(data), True)
    for literal in rule:
        index = index & bind_literal_index(data, literal.name, literal.op, literal.num_val, literal.nom_val)
    return index


def __unbound_rule_index(
        data,
        rule,
):
    """
    Find instances that are not covered by the rule

    Parameters
    ----------
    data : const double[::1, :]
        Input instances
    rule : list
        Input rule

    Returns
    -------
    ndarray
        Boolean index array for instances that are not covered by the rule
    """
    return ~_bind_rule_index(data, rule)


def unbound_rule_list_index(
        data,
        rules
):
    """
    Find instances that are not covered by the rule list

    Parameters
    ----------
    data : const double[::1, :]
        Input instances
    rules : list
        Input rule list

    Returns
    -------
    ndarray
        Boolean index array for instances that are not covered by the rule list
    """
    index = np.full(len(data), True)
    for rule in rules:
        index = index & __unbound_rule_index(data, rule)
    return index


def _bind_rule(
        data,
        rule
):
    """
    Find instances that are covered by the rule

    Parameters
    ----------
    data : const double[::1, :]
        Input instances
    rule : list
        Input rule

    Returns
    -------
    ndarray
        Instances that are covered by the rule
    """
    return data[_bind_rule_index(data, rule)]


def _unbound(
        data,
        rule
):
    """
    Find instances that are not covered by the rule

    Parameters
    ----------
    data : const double[::1, :]
        Input instances
    rule : list
        Input rule

    Returns
    -------
    ndarray
        Instances that are not covered by the rule
    """
    return data[__unbound_rule_index(data, rule)]


def _unbound_rule_list(
        data,
        rules
):
    """
    Find instances that are not covered by the rule list

    Parameters
    ----------
    data : const double[::1, :]
        Input instances
    rules : list
        Input rule list

    Returns
    -------
    ndarray
        Instances that are not covered by the rule list
    """
    return data[unbound_rule_list_index(data, rules)]


def _rule_predict(
        data,
        rule
):
    """
    Make a decision using input data and rule

    Parameters
    ----------
    data : const double[::1, :]
        Input instances
    rule : list
        Input rule

    Returns
    -------
    ndarray
        Predictions
    """
    if len(rule) == 0:
        return np.full(len(data), False)
    else:
        return _bind_rule_index(data, rule)


def _rule_list_predict(
        data,
        rules
):
    """
    Make a decision based on input data and rule list

    Parameters
    ----------
    data : const double[::1, :]
        Input instances
    rules : list
        Input rule list

    Returns
    -------
    ndarray
        Predictions
    """
    if len(rules) == 0:
        return np.full(len(data), False)
    else:
        return ~unbound_rule_list_index(data, rules)


def append_literal(
        rule,
        literal
):
    """
    Append a literal to a rule. It will delete unnecessary literals in that rule after append

    Parameters
    ----------
    rule : list
        Target rule
    literal : Literal
        New literal
    """
    # nominal variable will not be added twice
    if literal.op != EQ:
        # start from the last condition
        for i in range(len(rule) - 1, -1, -1):
            # delete redundant conditions
            if rule[i].name == literal.name:
                if rule[i].op == literal.op:
                    if literal.op == LE:
                        if literal.num_val < rule[i].num_val:
                            del rule[i]
                    else:
                        if literal.num_val > rule[i].num_val:
                            del rule[i]
    # add condition to the rule
    rule.append(literal)

