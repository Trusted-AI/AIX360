from aix360.algorithms.rbm import BooleanRuleCG as brcg

from ..trxf.core.dnf_ruleset import DnfRuleSet, Conjunction
from ..trxf.core.predicate import Predicate, Feature
from .utils import OPERATOR_MAPS


class BooleanRuleCG(brcg):
    """
    Same class as `aix360.algorithms.rbm.boolean_rule_cg.BooleanRuleCG`, the only 
    difference being that the explanations are available in the Technical Rule 
    Exchange Format (TRXF). 
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def explain(self, maxConj=None, prec=2) -> DnfRuleSet:
        """
        Overriden method that returns the BRCG learnt rules as a TRXF Ruleset.
        """
        # Selected conjunctions
        z = self.z.loc[:, self.w > 0.5]
        truncate = (maxConj is not None) and (z.shape[1] > maxConj)
        nConj = maxConj if truncate else z.shape[1]

        # Sort conjunctions by increasing order
        idxSort = z.sum().sort_values().index[:nConj]
        
        # Iterate over sorted conjunctions
        conj = [] # List of conjunctions
        for i in idxSort:
            pred_list = []
            
            # MultiIndex of features participating in rule i
            idxFeat = z.index[z[i] > 0]

            # String representations of features
            feats = idxFeat.get_level_values(0)
            ops = idxFeat.get_level_values(1)
            values = idxFeat.get_level_values(2).to_series()
                     
            for f, o, v in zip(feats, ops, values):

                if o in ['', 'not']: 
                    if "=" in f:
                        # Encoded feature
                        f, v = f.split("=")
                    else:
                        # boolean column
                        v = True # force values for boolean columms
                
                pred_list.append(Predicate(Feature(f), OPERATOR_MAPS[o], v))
            
            c = Conjunction(predicate_list=pred_list)
            conj.append(c)
        
        if self.CNF:
            then_part = 0 
        else:
            then_part = 1
        
        res = DnfRuleSet(conjunctions=conj,then_part=then_part)

        return res