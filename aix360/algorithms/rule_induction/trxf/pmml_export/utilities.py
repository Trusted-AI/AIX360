from typing import Dict

import numpy as np
import pandas as pd
from aix360.algorithms.rule_induction.trxf.pmml_export.models import Operator, SimplePredicate, CompoundPredicate, \
    BooleanOperator, Value

from aix360.algorithms.rule_induction.trxf.core import Conjunction, Relation

from aix360.algorithms.rule_induction.trxf.pmml_export import models


def extract_data_dictionary(X: pd.DataFrame, values: Dict):
    """
    Extract the data dictionary from a feature dataframe
    """
    dtypes = X.dtypes
    data_fields = []
    for index, value in dtypes.items():
        vals = None
        if np.issubdtype(value, np.integer):
            data_type = models.DataType.integer
            op_type = models.OpType.ordinal
        elif np.issubdtype(value, np.double):
            data_type = models.DataType.double
            op_type = models.OpType.continuous
        elif np.issubdtype(value, np.floating):
            data_type = models.DataType.float
            op_type = models.OpType.continuous
        elif np.issubdtype(value, np.bool_):
            data_type = models.DataType.boolean
            op_type = models.OpType.categorical
        else:
            data_type = models.DataType.string
            op_type = models.OpType.categorical
            vals = values[index] if values is not None and index in values else list(X[index].unique())
        wrapped_vals = list(map(lambda v: Value(v), vals)) if vals is not None else vals
        data_fields.append(models.DataField(name=str(index), optype=op_type, dataType=data_type, values=wrapped_vals))

    return models.DataDictionary(data_fields)


def trxf_to_pmml_predicate(trxf_conjunction: Conjunction):
    trxf_to_pmml_op = {
        Relation.EQ: Operator.equal,
        Relation.NEQ: Operator.notEqual,
        Relation.LT: Operator.lessThan,
        Relation.LE: Operator.lessOrEqual,
        Relation.GT: Operator.greaterThan,
        Relation.GE: Operator.greaterOrEqual
    }
    simple_predicates = [SimplePredicate(operator=trxf_to_pmml_op[trxf_predicate.relation],
                                         value=str(trxf_predicate.value),
                                         field=str(trxf_predicate.feature.variable_names[0]))
                         for trxf_predicate in trxf_conjunction.predicates]
    return CompoundPredicate(simplePredicates=simple_predicates, booleanOperator=BooleanOperator.and_)
