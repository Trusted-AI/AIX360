import math
import timeit
import numpy as np
from zoopt import Dimension, Objective, Parameter, Opt

def bb(x): #Black box function
    alpha = np.ones((len(x)))
    return alpha.dot(x)

def e(x): #Explanation function
    beta = 0.75*np.ones((len(x)))
    return beta.dot(x)

def f(x): #Fidelity function
    #fidelity = 1-abs(bb(x) - e(x))/max(abs(bb(x)), abs(e(x))) #Normalized MAE
    fidelity = 1-abs(bb(x) - e(x)) #MAE
    return fidelity

def UsampleBetRects(x,lb,ub,num):
    """ Sample uniformly between hyper-cubes [x-lb,x+lb] and [x-ub,x+ub]

    Returns:
        tuple: (boolean: Indicator of failure to sample in region, samples: array of shape `num x d`)
    """
    d = len(x)
    samples = np.zeros((num,d))
    totadded = 0
    if num <= 99:
        maxiter = 100 * num
    else:
        maxiter = 10 * num
    # maxiter = 10*num
    iters = 0

    while(totadded < num and iters <= maxiter):
        spl = (2*ub)*np.random.random((d,))+(x-ub)
        if(any(((spl[i] < x[i]-lb) or (spl[i] > x[i]+lb)) for i in range(d))):
            samples[totadded] = spl
            totadded += 1
        iters += 1

    if totadded == 0:
        print('Could not find any examples in between +-' + str(ub) + ' and ' + '+-' + str(lb) + ' around x = ' + str(x))
        return (True, samples)

    while(totadded < num):
        dup = np.random.randint(totadded)
        samples[totadded] = samples[dup]
        totadded += 1
    return (False, samples)

def GsampleBetRects(mu, sigma, x, lb, ub, num):
    """ Gaussian sampling between hyper-cubes [x-lb,x+lb] and [x-ub,x+ub]

    Returns:
       tuple: (boolean: Indicator of failure to sample in region, samples: array of shape `num x d`)
    """
    d = len(x)
    samples = np.zeros((num,d))
    totadded = 0
    maxiter = 10*num
    iters = 0

    samples[totadded] = mu
    totadded += 1
    #print('lb='+str(lb)+', ub='+str(ub))
    while(totadded < num and iters <= maxiter):
        spl = np.random.normal(mu, sigma)
        if(any(((spl[i] < x[i]-lb) or (spl[i] > x[i]+lb)) for i in range(d))):
            if(all(((spl[i] >= x[i]-ub) and (spl[i] <= x[i]+ub)) for i in range(d))):
                samples[totadded] = spl
                totadded += 1
        iters += 1
    #print('totadded='+str(totadded))
    while(totadded < num):
        dup = np.random.randint(totadded)
        samples[totadded] = samples[dup]
        totadded += 1
    return (False, samples)


def Certify_ZO(lb, ub, Q, lgQ, q, theta, x, sigma, s, f) -> tuple:
    """ Certification using ZO baseline (strategy=4 as used later)
    """
    dim_size = len(x)
    def func_wrapper(sol):
        x_item = sol.get_x()
        return f(np.asarray(x_item))

    # optimize between cubes [x-lb, x+lb] and [x-ub, x+ub]
    # hacky way to sample between cubes: have to iterate and ensure one dimension is outside inner cube
    totalruns = 10
    Q_1 = Q // 10

    min_fidelity = np.inf
    min_x = None

    if Q <= 10:
        print(f"setting totalrun to 1 to avoid `budget too small` error from zo!")
        totalruns = 1
        Q_1 = Q

    # iterate over dimensions to force one dimension outside `lb`
    for i_run in range(totalruns):
        try:
            rangetosample = []
            for m in range(dim_size):
                rangetosample.append([x[m] - ub, x[m] + ub])

            # pick one dimension and restrict (absolute) values between lb and ub
            i_run = np.random.randint(0, dim_size)

            # lb < |x| < ub => lb < x < ub and -ub < x < -lb (disjoint sets)
            # case 1: lb < x < ub
            rangetosample[i_run] = [x[i_run] + lb, x[i_run] + ub]
            dim = Dimension(dim_size, rangetosample, [True] * dim_size)
            obj = Objective(func_wrapper, dim)
            # perform optimization
            solution = Opt.min(obj, Parameter(budget=Q_1 / 2))

            if solution.get_value() <= min_fidelity:
                min_fidelity = solution.get_value()
                min_x = solution.get_x().copy()

            # case 2: -ub < x < -lb
            rangetosample[i_run] = [x[i_run] - ub, x[i_run] - lb]
            dim = Dimension(dim_size, rangetosample, [True] * dim_size)
            obj = Objective(func_wrapper, dim)

            # perform optimization
            solution = Opt.min(obj, Parameter(budget=Q_1 / 2))

            if solution.get_value() <= min_fidelity:
                min_fidelity = solution.get_value()
                min_x = solution.get_x().copy()
        except Exception as ex:
            print(f"***** Error in ZOOPT: {ex}")

    if min_x is None:
        return True, True, None

    if min_fidelity >= theta:
        # this region is certified
        return False, True, None

    # not certified, return violating example
    return False, False, np.asarray(min_x)


def Certify(lb, ub, Q, lgQ, q, theta, x, sigma, s, f, eps_fid=0.01, p=0.1):
    """
    Certify region between hypercubes [x-lb, x+lb] and [x-ub, x+ub] by choosing one of the three certification strategies given by `s`
    
    Returns:
        t1 (bool): Indicator of failure to sample in region
        t2 (bool): Whether region is certified or not
        min_fid_or_violator:
            If certified, minimum fidelity observed (float)
            If not certified, violating point ((d,) array)
            If failed to sample, np.zeros(d)
        prob_or_lcb (float):
            If certified and eps_fid is given, EVT probability that empirical minimum fidelity exceeds true minimum by more than eps_fid
            Else if certified and p is given, EVT lower confidence bound on true minimum fidelity (with probability 1-p)
            Otherwise None
    """
    d = len(x)
    if s==1: #Uniform Standard
        t, samples = UsampleBetRects(x,lb,ub,Q)
        if t:
            return (True, True, np.zeros(d), None)
        fids = []
        for i in range(Q):
            fid = f(samples[i])
            if fid < theta:
                return (False, False, samples[i], None)
            fids.append(fid)
        # if min(fids) < theta:
        #     # print(f"returning min fid={min(fids)}, s={samples[fids.index(min(fids))]}, {f(samples[fids.index(min(fids))])}")
        #     return (False, False, samples[fids.index(min(fids))])
        
        # Certified, compute failure probability or lower confidence bound on true minimum fidelity
        if eps_fid is not None:
            prob_or_lcb = compute_EVT_fail_prob(fids, eps_fid, d)
        elif p is not None:
            prob_or_lcb = compute_EVT_lcb(fids, p, d)
        else:
            prob_or_lcb = None
        return (False, True, min(fids), prob_or_lcb)

    elif s==2: #Uniform Incremental
        fllgQ = int(np.floor(lgQ))
        fids = []
        for i in range(fllgQ):
            n = min(2**i,q)
            # print(f"\ti:{i + 1}/{fllgQ}, sampling {n} prototypes")
            t, prototypes = UsampleBetRects(x,lb,ub,n)
            if t:
                return (True, True, np.zeros(d), None)
            for j in range(n):
                num = int(np.floor(q/n))
                t, samples = GsampleBetRects(prototypes[j],sigma,x,lb,ub,num)
                if t:
                    return (True, True, np.zeros(d), None)
                for k in range(num):
                    fids.append(f(samples[k]))
                    if f(samples[k]) < theta:
                        return (False, False, samples[k], None)
                # if min(fids) < theta:
                #     # print(f"returning min fid={min(fids)}, {f(samples[fids.index(min(fids))])}")
                #     return (False, False, samples[fids.index(min(fids))])
        return (False, True, min(fids), 1)
    
    elif s==5: # an i.i.d. version of Uniform Incremental to enable EVT lower confidence bound
        d = len(x)
        rng = np.random.default_rng()
        
        # Number of Gaussian mixtures
        lgQ_rounded = round(lgQ)  # could be floor instead of round
        # Sample mixture indices uniformly
        ind_mixture = rng.integers(lgQ_rounded, size=Q)

        fids = []
        # Iterate over Gaussian mixtures
        for i in range(lgQ_rounded):
            # Number of samples from this mixture
            q_mixture = (ind_mixture == i).sum()
            # Number of mixture components/prototypes
            n = min(2**i, q_mixture)
            
            # Sample prototypes
            t, prototypes = UsampleBetRects(x, lb, ub, n)
            if t:
                return (True, True, np.zeros(d), None)
            # Sample component indices uniformly
            ind_component = rng.integers(n, size=q_mixture)
            
            # Iterate over mixture components/prototypes
            for j in range(n):
                # Number of samples from this mixture component
                num = (ind_component == j).sum()
                if num == 0:
                    continue
                # Sample from Gaussian centered at prototype
                t, samples = GsampleBetRects(prototypes[j], sigma, x, lb, ub, num)
                if t:
                    return (True, True, np.zeros(d), None)
                
                # Iterate over samples from this mixture component
                for k in range(num):
                    fid = f(samples[k])
                    if fid < theta:
                        return (False, False, samples[k], None)
                    fids.append(fid)
        
        # Certified, compute failure probability or lower confidence bound on true minimum fidelity
        if eps_fid is not None:
            prob_or_lcb = compute_EVT_fail_prob(fids, eps_fid, d)
        elif p is not None:
            prob_or_lcb = compute_EVT_lcb(fids, p, d)
        else:
            prob_or_lcb = None
        return (False, True, min(fids), prob_or_lcb)

    elif s==3: #Adaptive Incremental
        fllgQ = int(np.floor(lgQ))
        fids = []
        for i in range(fllgQ):
            if i*2**i <= q:
                n, k = 2**i, i
            else:
                n = 2**k
            t, prototypes = UsampleBetRects(x,lb,ub,n)
            if t:
                return (True, True, np.zeros(d), None)
            clgn = int(np.ceil(math.log2(n)))
            m = n
            for j in range(clgn):
                minfidvalues = np.zeros(m)
                num = int(np.floor(q/(m*clgn)))
                for p in range(m):
                    t, samples = GsampleBetRects(prototypes[p],sigma,x,lb,ub,num)
                    if t:
                        return (True, True, np.zeros(d), None)
                    minfidval = np.Inf
                    for l in range(num):
                        fids.append(f(samples[l]))
                        if(f(samples[l]) < minfidval):
                            if(f(samples[l]) < theta):
                                return (False, False, samples[l], None)
                            minfidval = f(samples[l])
                    minfidvalues[p] = minfidval
                sortedfididx = np.argsort(minfidvalues)
                prototypes = prototypes[sortedfididx]
                m = int(np.ceil(m/2))
        return (False, True, min(fids), 1)

    elif s==4:
        return *Certify_ZO(lb, ub, Q, lgQ, q, theta, x, sigma, s, f), 1.0  # return 1.0 as EVT expects 4 tuple return object

def compute_EVT_lcb(fids, p, d):
    """
    Compute EVT lower confidence bound on true minimum fidelity

    Parameters
    ----------
    fids : list
        Sampled fidelity values
    p : float
        Lower confidence bound is less than true minimum fidelity with probability 1-p
    d : int
        Dimension of input x

    Returns
    -------
    lcb : float
        Lower confidence bound
    """
    fids_sorted = np.sort(fids)
    # Lower confidence bound = lowest fidelity - half-width proportional to gap between lowest and second-lowest
    lcb = fids_sorted[0] - (fids_sorted[1] - fids_sorted[0]) / ((1-p)**(-2/d) - 1)
    print("empirical minimum =", fids_sorted[0])
    print("confidence interval half-width =", fids_sorted[0] - lcb)
    
    return lcb

def compute_EVT_fail_prob(fids, eps, d):
    """
    Compute EVT probability that empirical minimum fidelity exceeds true minimum by more than epsilon

    Parameters
    ----------
    fids : list
        Sampled fidelity values
    eps : float
        Tolerance epsilon
    d : int
        Dimension of input x

    Returns
    -------
    p : float
        Probability that empirical minimum exceeds true minimum by more than epsilon
    """
    fids_sorted = np.sort(fids)
    p = 1 - (1 + (fids_sorted[1] - fids_sorted[0]) / eps)**(-d/2)
    # print("gap between smallest and second-smallest =", fids_sorted[1] - fids_sorted[0])
    
    return p

#example to be certified x, black-box fidelity function f (.) (≥ 0), minimum fidelity
#threshold theta, number of iterations (or regions to check) Z, query budget per region Q, lower bound half-width
#(lb), upper bound half-width (ub) and certification strategy to use (s=1 (unif), 2 (uI), 3 (aI), 4 (ZO)

def Ecertify(x, theta, Z, Q, lb=0, ub=np.Inf, sigma_0=0.1, s=1, quality=f, choice="min", eps_mul=0.1, eps_fid=0.01):
    """ main function to expose to be used by the wrapper class.

    Args:
    -----
        x (np.array): example to be certified
        theta (float): minimum fidelity threshold theta
        Z (int): number of iterations (or regions to check) Z
        Q (int): query budget per region Q
        lb (int, optional): lower bound half-width. Defaults to 0.
        ub (_type_, optional): upper bound half-width. Defaults to np.Inf.
        sigma_0 (float, optional): std deviation of gaussian for unifI and adaptI strategies. Defaults to 0.1.
        s (int, optional):certification strategy to use (s=1 (unif), 2 (uI), 3 (aI), 4 (ZO). Defaults to 1.
        quality (callable, optional): black-box fidelity function f (.) (≥ 0). Defaults to f.
        choice (str, optional): _description_. Defaults to "min".
        eps_mul (float, optional): spatial resolution. Defaults to 0.1.
        eps_fid (float, optional): _description_. Defaults to 0.01.

    Raises:
    -------
        ValueError: _description_

    Returns:
    --------
        tuple: (Currbst: certification width w, prob_or_lcb)
    """
    Currbst = 0.0
    min_fid_curr = np.inf
    prob_or_lcb = 1
    d = len(x)
    eps = eps_mul/d
    B = np.Inf
    init_fidelity = quality(x)
    # print(f"fidelity at x0: f(x)={init_fidelity:.4f}")
    if init_fidelity < theta:
        return -1
    lgQ = math.log2(Q)
    q = int(np.floor(Q/lgQ))
    for z in range(Z):
        # print(f"\nz: {z} curbst: {Currbst:.4f} lb: {lb:.4f} ub: {ub:.4f}")
        if ub-lb < eps:
            # print("returning because lowest spatial resolution to reached (eps)!")
            return Currbst, prob_or_lcb
        sigma = sigma_0 #* (ub-lb)/d
        t1,t2, min_fid_or_violator, prob_or_lcb_new = Certify(lb, ub, Q, lgQ, q, theta, x, sigma, s, quality, eps_fid=eps_fid)
        # print(f"\t\t b = {b}")
        if t1:
            return Currbst, prob_or_lcb
        if t2:
            # t2 => this region is certified, need to double the search space
            Currbst, lb = ub, ub
            ub = min((B+ub)/2, 2*ub)
            # if min_fid_or_violator < min_fid_curr:
            #     # Update minimum fidelity and prob_or_lcb
            #     min_fid_curr = min_fid_or_violator
            #     # Convert into probability of being within eps_fid
            #     prob_or_lcb = 1 - prob_or_lcb_new
            # print(f"\tcertified! doubling ub to {ub:.4f}, with B={B:.4f}")
        else:
            # this region is not certified and a violator has been found, need to halve search space and reset B
            temp = []
            for i in range(d):
                if abs(min_fid_or_violator[i]-x[i]) > lb:
                    temp.append(abs(min_fid_or_violator[i]-x[i]))

            # try min, max or average of the violator radius --- this is another hyperparameter and depends on d
            if choice == "min":
                B = min(temp)
            elif choice == "max":
                B = max(temp)
            elif choice == "mean":
                B = sum(temp)/len(temp)
            else:
                raise ValueError('choice must be "min", "max" or "mean"!')

            ub = (B+lb)/2
            # print(f"\tnot certified! halving ub to {ub:.4f}, changed B={B:.4f}")
    return Currbst, prob_or_lcb


if __name__ == '__main__':
    #Calling Function
    d = 5 #input dimensionality
    x = np.array([0.0]*d) #input
    theta = 0.75 #fidelity threshold
    Z = 10 #number of hypercubes to certify
    Q = 1000 #query budget for each hypercube
    s = 3 #strategy 1: unif, 2:unifI, 3:adaptI, 4:zoopt, 5:unifI_soft
    eps = 0.1/d #min gap between lb and ub
    numruns = 10

    certicubeperrun = np.zeros(numruns)
    evtprobs = np.zeros(numruns)
    t_0 = timeit.default_timer()
    for i in range(numruns):
        ub = 1  # initial hypercube half-width
        lb = 0  # since x is the center of the hypercube
        Currbst = 0  # current certified hypercube half width
        Certicube = Ecertify(x, theta, Z, Q, lb, ub, 0.1, s, f, choice=np.random.choice(["min", "max", "mean"]))
        certicubeperrun[i] = Certicube[0]
        evtprobs[i] = Certicube[1]
    t_1 = timeit.default_timer()
    time_per_run = round((t_1 - t_0)/numruns, 3)

    print('Certified half-width around input x = ' + str(x) + ' using strategy ' + str(s) + ' for theta = ' + str(theta) + ' is ')
    #print(str(np.mean(certicubeperrun)) + ' +- ' + str(np.std(certicubeperrun)/np.sqrt(numruns)))
    print(f"found: {np.mean(certicubeperrun):.4f} +- {np.std(certicubeperrun)/np.sqrt(numruns):.6f}, true: {1/d:.4f}")
    print(f"EVT lb: {np.mean(evtprobs):.4f} +- {np.std(evtprobs)/np.sqrt(numruns):.6f}")
    print(f"\nTime per run: {time_per_run} s")