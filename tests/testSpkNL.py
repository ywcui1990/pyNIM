__author__ = 'ywcui'

import spkNL
reload(spkNL)
from spkNL import *
import numpy as np
from matplotlib import pyplot as plt
from pylab import ion
ion()

nlFun = lambda g, params: params[0] * np.exp( params[1] * (g-params[2]))
params = np.array([.1,1,0])

g = np.random.randn(10000)
r = nlFun(g, params)
spk = np.random.poisson(lam = r)

# generating signal at spike timing
gspk = g[spk>0]

nl = spkNL('exp')

# estimate spiking NL using histogram method
estnl = nl.histEstimate(g, gspk, display = 0)

# fit spiking NL

# unconstrained optimization
nl.fitSpkNL(g, spk, hold_const=[])

# nl.fitSpkNL(g, spk)

