import numpy as np
import pandas as pd


class PricingInstance(object):
    """Instance of the pricing problem"""
    def __init__(self, rp, rn, Xp, Xn, v0, z0):
        self.rp = rp
        self.rn = rn
        self.Xp = Xp
        self.Xn = Xn
        self.v0 = v0
        self.z0 = z0
    
    def eval_singletons(self, lambda1):
        """Evaluate all singleton solutions (adding one literal)"""
        self.Rp = np.dot(self.rp, self.Xp)
        self.Rn = np.dot(self.rn, self.Xn)
        self.v1 = self.v0 - self.Rp - self.Rn + lambda1
        self.v1 = pd.Series(self.v1, index=self.Xp.columns)
        
    def compute_LB(self, lambda1):
        """Compute lower bound on higher-order solutions"""
        Rp0 = self.rp.sum()
        if np.ndim(lambda1):
            self.LB = np.array([])
        else:
            self.LB = np.minimum(np.cumsum(np.sort(self.Rp)[::-1])[1:], Rp0)
            self.LB += np.sort(self.Rn)[-2::-1]
            self.LB -= lambda1 * np.arange(2, len(self.Rp)+1)
            self.LB = self.v0 - self.LB

        # Lower bound specific to each singleton solution
        self.LB1 = self.v1 + self.Rp - Rp0 + lambda1
        if len(self.LB):
            self.LB1[self.LB1 < self.LB.min()] = self.LB.min()


def beam_search_K1(r, X, lambda0, lambda1, UB=0, D=10, B=5, wLB=0.5, eps=1e-6, stopEarly=True):
    """Beam search to generate SINGLE SOLUTION (K = 1) to pricing problem
    Problem parameters:
        r = cost vector (residuals)
        X = binary feature DataFrame
        lambda0 = fixed cost of a term
        lambda1 = cost per literal
    
    Algorithm parameters:
        UB = initial upper bound on value of solutions
        D = maximum degree
        B = beam width
        wLB = weight on lower bound in evaluating nodes
        eps = numerical tolerance on comparisons
        stopEarly = stop after current degree once solution is found
    """

    # Initialize output
    vOut = np.array([])
    zOut = pd.Series(index=X.columns)

    # Initialize queue with root instance
    # Separate data according to positive and negative residuals
    rp = r[r > 0]
    rn = r[r < 0]
    Xp = 1 - X.loc[r > 0]
    Xn = 1 - X.loc[r < 0]
    instCurr = [PricingInstance(rp, rn, Xp, Xn, r.sum() + lambda0, pd.Series(0, index=zOut.index))]
    
    # Iterate over increasing degree while queue is non-empty
    deg = 0
    while (not len(vOut) or not stopEarly) and len(instCurr) and deg < D:
        deg += 1

        # Initialize list of children to process
        vNext = np.array([])
        vNextMax = np.inf
        zNext = pd.DataFrame([], index=X.columns)
        idxInstNext = np.array([], dtype=int)
        idxFeatNext = np.array([])
        
        # Process instances in queue
        for (idxInst, inst) in enumerate(instCurr):

            # Evaluate all singleton solutions
            inst.eval_singletons(lambda1)

            # Solutions that improve on current output
            vCand = inst.v1[inst.v1 < UB - eps]
            if len(vCand):
                # Update output with best of these solutions
                idxMin = vCand.idxmin()
                UB = vCand[idxMin]
                vOut = np.array([UB])
                zOut = inst.z0.copy()
                zOut[idxMin] = 1
                
            # Compute lower bounds on higher-degree solutions
            inst.compute_LB(lambda1)    
            # Evaluate children using weighted average of their costs and LBs
            vChild = (1 - wLB) * inst.v1 + wLB * inst.LB1

            # Best children with potential to improve on current output and current candidates (allow for duplicate removal)
            vChild = vChild[(inst.LB1 < UB - eps) & (vChild < vNextMax - eps)].sort_values()[:B+idxInst]
            if len(vChild):
                # Feature indicators of these best children
                zChild = pd.DataFrame(zOut.index.values[:,np.newaxis] == vChild.index.values, index=zOut.index).astype(int)
                zChild = zChild.add(inst.z0, axis=0)
                
                # Append to current candidates
                vNext = np.append(vNext, vChild.values)
                zNext = pd.concat([zNext, zChild], axis=1, ignore_index=True)
                idxInstNext = np.append(idxInstNext, np.full(B+idxInst, idxInst))
                idxFeatNext = np.append(idxFeatNext, vChild.index.values)
                # Remove duplicates
                _, idxUniq = np.unique(zNext, return_index=True, axis=1)
                vNext = vNext[idxUniq]
                zNext = zNext.iloc[:,idxUniq]
                idxInstNext = idxInstNext[idxUniq]
                idxFeatNext = idxFeatNext[idxUniq]
                # Update candidates
                idxBest = np.argsort(vNext)[:B]
                vNext = vNext[idxBest]
                if len(vNext):
                    vNextMax = vNext[-1]
                zNext = zNext.iloc[:,idxBest]
                zNext.columns = range(zNext.shape[1])
                idxInstNext = idxInstNext[idxBest]
                idxFeatNext = idxFeatNext[idxBest]
                
        # Instances to process in next iteration
        instNext = []
        for (idxInst, i, idxz) in zip(idxInstNext, idxFeatNext, zNext):
            # Create pricing instance
            # Remove covered rows
            rowKeep = instCurr[idxInst].Xp[i] == 0
            rp = instCurr[idxInst].rp[rowKeep]
            Xp = instCurr[idxInst].Xp.loc[rowKeep]
            rowKeep = instCurr[idxInst].Xn[i] == 0
            rn = instCurr[idxInst].rn[rowKeep]
            Xn = instCurr[idxInst].Xn.loc[rowKeep]
            # Remove redundant features
            if type(Xp.columns) is pd.MultiIndex: 
                colKeep = pd.Series(Xp.columns.get_level_values(0) != i[0], index=Xp.columns)
                if i[1] == '<=':
                    thresh = Xp[i[0]].columns.get_level_values(1).to_series().replace('NaN', np.nan).values
                    colKeep[i[0]] = (Xp[i[0]].columns.get_level_values(0) == '>') & (thresh < i[2])
                elif i[1] == '>':
                    thresh = Xp[i[0]].columns.get_level_values(1).to_series().replace('NaN', np.nan).values
                    colKeep[i[0]] = (Xp[i[0]].columns.get_level_values(0) == '<=') & (thresh > i[2])
                elif i[1] == '!=':
                    colKeep[i[0]] = (Xp[i[0]].columns.get_level_values(0) == '!=') & (Xp[i[0]].columns.get_level_values(1) != i[2])
                Xp = Xp.loc[:, colKeep]
                Xn = Xn.loc[:, colKeep]
            instNext.append(PricingInstance(rp, rn, Xp, Xn, instCurr[idxInst].v1[i], zNext[idxz]))

        instCurr = instNext
        
    # Conjunctions corresponding to solutions
    if zOut.count():
        zOut = pd.DataFrame(zOut)
    else:
        zOut = pd.DataFrame(index=X.columns)

    aOut = 1 - (np.dot(1 - X, zOut) > 0)
    
    return vOut, zOut, aOut


def beam_search(r, X, lambda0, lambda1, K=1, UB=0, D=10, B=5, wLB=0.5, eps=1e-6, stopEarly=False):
    """Beam search to generate solutions to pricing problem
    Problem parameters:
        r = cost vector (residuals)
        X = binary feature DataFrame
        lambda0 = fixed cost of a term
        lambda1 = cost per literal
    
    Algorithm parameters:
        K = maximum number of solutions returned
        UB = initial upper bound on value of solutions
        D = maximum degree
        B = beam width
        wLB = weight on lower bound in evaluating nodes
        eps = numerical tolerance on comparisons
        stopEarly = stop after current degree once solution is found
    """

    # Initialize output
    vOut = np.array([])
    zOut = pd.DataFrame(index=X.columns)

    # Remove redundant rows by grouping by unique feature combinations and summing residual
    X2 = X.copy()
    r2 = r
    
    # Initialize queue with root instance
    # Separate data according to positive and negative residuals
    rp = r2[r2 > 0]
    rn = r2[r2 < 0]
    Xp = 1 - X2.loc[r2 > 0]
    Xn = 1 - X2.loc[r2 < 0]
    instCurr = [PricingInstance(rp, rn, Xp, Xn, r2.sum() + lambda0, pd.Series(0, index=zOut.index))]
    
    # Iterate over increasing degree while queue is non-empty
    deg = 0
    while (not len(vOut) or not stopEarly) and len(instCurr) and deg < D:
        deg += 1

        # Initialize list of children to process
        vNext = np.array([])
        vNextMax = np.inf
        zNext = pd.DataFrame([], index=X2.columns)
        idxInstNext = np.array([], dtype=int)
        idxFeatNext = np.array([])
        
        # Process instances in queue
        for (idxInst, inst) in enumerate(instCurr):

            # Evaluate all singleton solutions
            inst.eval_singletons(lambda1)

            # Best solutions that also improve on current output (allow for duplicate removal)
            vCand = inst.v1[inst.v1 < UB - eps].sort_values()[:K+B]
            if len(vCand):
                zCand = pd.DataFrame(zOut.index.values[:,np.newaxis] == vCand.index.values, index=zOut.index).astype(int)
                zCand = zCand.add(inst.z0, axis=0)
                # Append to current output
                vOut = np.append(vOut, vCand.values)
                zOut = pd.concat([zOut, zCand], axis=1, ignore_index=True)
                # Remove duplicates
                _, idxUniq = np.unique(zOut, return_index=True, axis=1)
                vOut = vOut[idxUniq]
                zOut = zOut.iloc[:,idxUniq]
                # Update output
                indBest = np.argsort(vOut)[:K]
                vOut = vOut[indBest]
                UB = vOut.max()
                zOut = zOut.iloc[:,indBest]
                zOut.columns = range(zOut.shape[1])
            
            # Compute lower bounds on higher-degree solutions
            inst.compute_LB(lambda1)
            # Evaluate children using weighted average of their costs and LBs
            vChild = (1 - wLB) * inst.v1 + wLB * inst.LB1
            
            # Best children with potential to improve on current output and current candidates (allow for duplicate removal)
            vChild = vChild[(inst.LB1 < UB - eps) & (vChild < vNextMax - eps)].sort_values()[:B+idxInst]
            if len(vChild):
                # Feature indicators of these best children
                zChild = pd.DataFrame(zOut.index.values[:,np.newaxis] == vChild.index.values, index=zOut.index).astype(int)
                zChild = zChild.add(inst.z0, axis=0)
                
                # Append to current candidates
                vNext = np.append(vNext, vChild.values)
                zNext = pd.concat([zNext, zChild], axis=1, ignore_index=True)
                idxInstNext = np.append(idxInstNext, np.full(B+idxInst, idxInst))
                idxFeatNext = np.append(idxFeatNext, vChild.index.values)
                # Remove duplicates
                _, idxUniq = np.unique(zNext, return_index=True, axis=1)
                vNext = vNext[idxUniq]
                zNext = zNext.iloc[:,idxUniq]
                idxInstNext = idxInstNext[idxUniq]
                idxFeatNext = idxFeatNext[idxUniq]
                # Update candidates
                idxBest = np.argsort(vNext)[:B]
                vNext = vNext[idxBest]
                if len(vNext):
                    vNextMax = vNext[-1]
                zNext = zNext.iloc[:,idxBest]
                zNext.columns = range(zNext.shape[1])
                idxInstNext = idxInstNext[idxBest]
                idxFeatNext = idxFeatNext[idxBest]
                
        # Instances to process in next iteration
        instNext = []
        for (idxInst, i, idxz) in zip(idxInstNext, idxFeatNext, zNext):
            # Create pricing instance
            # Remove covered rows
            rowKeep = instCurr[idxInst].Xp[i] == 0
            rp = instCurr[idxInst].rp[rowKeep]
            Xp = instCurr[idxInst].Xp.loc[rowKeep]
            rowKeep = instCurr[idxInst].Xn[i] == 0
            rn = instCurr[idxInst].rn[rowKeep]
            Xn = instCurr[idxInst].Xn.loc[rowKeep]
            # Remove redundant features
            colKeep = pd.Series(Xp.columns.get_level_values(0) != i[0], index=Xp.columns)
            if i[1] == '<=':
                thresh = Xp[i[0]].columns.get_level_values(1).to_series().replace('NaN', np.nan).values
                colKeep[i[0]] = (Xp[i[0]].columns.get_level_values(0) == '>') & (thresh < i[2])
            elif i[1] == '>':
                thresh = Xp[i[0]].columns.get_level_values(1).to_series().replace('NaN', np.nan).values
                colKeep[i[0]] = (Xp[i[0]].columns.get_level_values(0) == '<=') & (thresh > i[2])
            elif i[1] == '!=':
                colKeep[i[0]] = (Xp[i[0]].columns.get_level_values(0) == '!=') & (Xp[i[0]].columns.get_level_values(1) != i[2])
            Xp = Xp.loc[:, colKeep]
            Xn = Xn.loc[:, colKeep]
            instNext.append(PricingInstance(rp, rn, Xp, Xn, instCurr[idxInst].v1[i], zNext[idxz]))

        instCurr = instNext
        
    # Conjunctions corresponding to solutions
    aOut = 1 - (np.dot(1 - X, zOut) > 0)

    return vOut, zOut, aOut
