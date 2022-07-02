
from ..trxf.core.predicate import Relation

OPERATOR_MAPS = {
    '>=': Relation.GE,
    '>' : Relation.GT,
    '<=': Relation.LE,
    '<' : Relation.LT,
    '==': Relation.EQ,
    '!=': Relation.NEQ,
    'not': Relation.NEQ,    # For boolean columns
    '': Relation.EQ,        # For boolean columns
}