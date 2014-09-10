__author__ = 'ywcui'

import numpy as np

def calculate_generating_signal(kvec, X, g0=None, gmult=None, offsetlabel=None, modelstruct=None, FitGain=0):

    Npar = len(kvec)
    (NT, NK) = X.shape

    # additive signal
    if g0 is None:
        g0 = np.zeros(NT, dtype='float32')

    # unpack kvec
    if FitGain:
        gain = kvec[NK]
        b = kvec[NK+1:]
    else:
        gain = 1
        b = kvec[NK:]
    k = kvec[:NK]

    if modelstruct is None:
        Ntar = 1
        Klen = [NK]
    else:
        Ntar = modelstruct['Nmod']
        Klen = modelstruct['Klen']

    if offsetlabel is None:
        offsetlabel = [np.arange(start=0,stop=NT)]

    if gmult is None:
        g = np.dot(X, k)
    else:
        # if multiplicative signal exist
        koffset = 0
        g = np.zeros((NT,1),dtype='float32')
        for i in range(Ntar):
            krange = koffset + np.arange(start=0, stop=Klen[i])
            g += np.dot(X[:,krange], k[krange]) * gmult[i]
            koffset += Klen[i]

        # linear terms with no multiplicative signal
        glin = 0
        for i in range(koffset, NK):
            glin += k[i] * X[:,i]
        g = g + glin

    g += gain*g0

    # include offset into generating signal
    for i in range(len(offsetlabel)):
        g[offsetlabel[i]] = g[offsetlabel[i]] + b[i]

    return g


def llexp(kvec, X, R, g0=None, gmult=None, spkNL=None, offsetlabel=None, modelstruct=None, FitGain=0):

    Npar = len(kvec)
    (NT, NK) = X.shape

    if offsetlabel is None:
        offsetlabel = [np.arange(start=0,stop=NT)]

    g = calculate_generating_signal(kvec, X, g0, gmult=None, offsetlabel=None, modelstruct=None, FitGain=0)

    alpha, beta, theta = 1,1,0
    if spkNL is None:
        NLtype = 'eLog'
    else:
        NLtype = spkNL.type
        if spkNL.params is not []:
            alpha, beta, theta = spkNL.params


    # calculate generating signal

    if NLtype is 'exp':
        g[g > 10] = 10
        r = np.exp[g]
        rtot = np.sum(r)
        residue = R - r
        ll = np.sum(R*r) - rtot

    elif NLtype is 'eLog':
        g -= theta
        g[g > 10] = 10
        ekx = np.exp(beta*g)
        r = alpha*np.log(1+ekx)
        r[r < 1e-10] = 1e-10

        ll = np.sum(np.log(r)*R) - np.sum(r)

    if NLtype is 'none':
        # if NLtype is none, we are likely modeling intracellular current
        # normalize by length of recording in this case
        llnorm = NT
    else:
        # extracellular recording (spikes)
        # normalize LL by spikes
        llnorm = np.sum(R)

    ll = -ll/llnorm
    return ll


def llexp_deriv(kvec, X, R, g0=None, gmult=None, spkNL=None, offsetlabel=None, modelstruct=None, FitGain=0):

    Npar = len(kvec)
    (NT, NK) = X.shape

    if modelstruct is None:
        Ntar = 1
        Klen = [NK]
    else:
        Ntar = modelstruct['Nmod']
        Klen = modelstruct['Klen']

    if offsetlabel is None:
        offsetlabel = [np.arange(start=0,stop=NT)]

    # additive signal
    if g0 is None:
        g0 = np.zeros(NT, dtype='float32')


    g = calculate_generating_signal(kvec, X, g0, gmult=None, offsetlabel=None, modelstruct=None, FitGain=0)

    alpha, beta, theta = 1,1,0
    if spkNL is None:
        NLtype = 'eLog'
    else:
        NLtype = spkNL.type
        if spkNL.params is not []:
            alpha, beta, theta = spkNL.params


    # calculate generating signal
    grad = np.zeros(Npar,dtype='float32')

    if NLtype is 'exp':
        g[g > 10] = 10
        r = np.exp[g]
        residue = R - r

        if gmult is None:
            grad[:NK] = np.dot(np.transpose(residue), X)
        else:
            koffset = 0
            for i in range(Ntar):
                krange = koffset + np.arange(start=0, end=Klen[i])
                grad[krange] = np.dot(np.transpose(residue*gmult[i]), X)
                koffset += Klen[i]

        # gradient w.r.t. gain terms on g0
        offseti = NK
        if FitGain:
            grad[offseti] = np.dot(np.transpose(residue), g0)
            offseti += 1

        # gradient w.r.t. threshold
        for i in range(len(offsetlabel)):
            grad[offseti] = np.sum(residue[offsetlabel[i]])

    elif NLtype is 'eLog':
        g -= theta
        g[g > 10] = 10
        ekx = np.exp(beta*g)
        r = alpha*np.log(1+ekx)
        r[r < 1e-10] = 1e-10

        wg = alpha*beta*ekx/(1+ekx)
        ers = R*wg/r
        if gmult is None:
            grad[:NK] = np.dot(np.transpose(ers-wg), X)
            grad = np.transpose(grad)
        else:
            koffset = 0
            for i in range(Ntar):
                krange = koffset + np.arange(start=0, end=Klen[i])
                grad[krange] = np.dot(np.transpose((ers-wg)*gmult[i]), X)
                koffset += Klen[i]

        # gradient w.r.t. gain terms on g0
        offseti = NK
        if FitGain:
            grad[offseti] = np.dot(np.transpose(ers-wg), g0)
            offseti += 1

        # gradient w.r.t. threshold
        for i in range(len(offsetlabel)):
            grad[offseti] = np.sum(ers[offsetlabel[i]]) - np.sum(wg[offsetlabel[i]])

    if NLtype is 'none':
        # if NLtype is none, we are likely modeling intracellular current
        # normalize by length of recording in this case
        llnorm = NT
    else:
        # extracellular recording (spikes)
        # normalize LL by spikes
        llnorm = np.sum(R)

    grad = -grad/llnorm
    return grad


def llSpkNL(params, g, robs, nlType='eLog'):
    """
    Calculate Log-likelihood for a given set of parameters

    :param params: parameters of the spiking non-linearity
    :param g: generating signal
    :param robs: Observed response (binned spike count)
    :param nlType: type of nonlinearity
    :return: Log-likelihood
    """

    numspks = sum(robs) # number of spikes

    internal = params[1]*(g - params[2])
    if nlType == 'eLog':
        lexp = np.log(1 + np.exp(internal))
        r = params[0] * lexp
    elif nlType == 'exp':
        expint = np.exp(internal)
        r = params[0] * expint

    r[r < 1e-10] = 1e-10

    ll = -np.sum(robs*np.log(r)-r)/numspks

    return ll

def llSpkNL_deriv(params, g, robs, nlType='eLog'):

    numspks = sum(robs) # number of spikes

    internal = params[1]*(g - params[2])

    if nlType == 'eLog':
        lexp = np.log(1 + np.exp(internal))
        r = params[0] * lexp
    elif nlType == 'exp':
        expint = np.exp(internal)
        r = params[0] * expint

    r[r < 1e-10] = 1e-10

    multfrac = (robs/r -1)

    grad = np.zeros((3),dtype='float32')
    if nlType == 'eLog':
        fract = np.exp(internal)/(1 + np.exp(internal))
        fract[internal > 50] = 1
        grad[0] = np.dot(multfrac, lexp)
        grad[1] = np.dot(multfrac, params[0]*(fract*(g-params[2])))
        grad[2] = np.dot(multfrac, -params[0]*params[1]*fract)
    elif nlType == 'exp':
        grad[0] = np.dot(multfrac, expint)
        grad[1] = np.dot(multfrac, params[0]*(expint*(g-params[2])))
        grad[2] = np.dot(multfrac, -params[0]*params[1]*(expint*(g-params[2])))

    grad = -grad/numspks
    return grad