__author__ = 'ywcui'
"""
This script test the class intNL
The true underlying internal NL is a rectified-linear NL
Response are simulated with the internal NL and spiking NL
The intNl class is used to model this response, the goal is to
recover the true underlying internal NL
"""

import intNL
reload(intNL)
from intNL import *
import numpy as np
from matplotlib import pyplot as plt
from pylab import ion
import copy
ion()

import costfunction
reload(costfunction)
from costfunction import llexp, llexp_deriv

def recNL(x):
    y = copy.deepcopy(x)
    y[x < 0] = 0
    return y

params = np.array([.1,1,0])
nlFunSpkNL = lambda g, params: params[0] * np.exp( params[1] * (g-params[2]))

# stimulus
s = np.random.standard_normal(size=100000)*0.5
# generating signal
g = recNL(s)
# firing rate
r = nlFunSpkNL(g, params)
# simulated spikes
spk = np.random.poisson(lam = r)

# initialize the intNL class
nl = intNL(type='nonparam')

# adjust the x-axis of the NL to cover the range of stimulus
nl.adjustScale(s)

(X,k) = nl.makeXmatrix(s)

# test the equivalence of X.k and process function
gs1 = np.dot(X,k)
gs2 = nl.process(s)
assert(np.max(np.abs(gs1-gs2))<1e-6)

k = np.append(k, 0)

# test of the gradient function
llinit = llexp(k, X, spk)
grad = llexp_deriv(k, X, spk)
llstep = 1e-7
gradest = np.zeros_like(grad)
for i in range(len(k)):
    kprobe = copy.copy(k)
    kprobe[i] += llstep
    llprobe = llexp(kprobe, X, spk)
    gradest[i] = (llprobe - llinit)/llstep

assert(np.max(np.abs(gradest-grad))<1e-4)

# optimize coefficients for the
from scipy.optimize import minimize
res = minimize(llexp, k, args = (X, spk),\
               jac = llexp_deriv, method = 'BFGS', options = {'disp':True})

knl = res.x[1:-1]
plt.close('all')
plt.plot(knl)