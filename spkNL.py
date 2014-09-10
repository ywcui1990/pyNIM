__author__ = 'ywcui'

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import copy
from costfunction import llSpkNL, llSpkNL_deriv

class spkNL:

    "Class implementing the spike non-linearity"

    def __init__(self,
                 nlType = 'exp',
                 params = [1,1,0],
                 ):
        """
        Construct the spkNL

        @param type Type of spiking nonlinearity, currently support 'exp', 'eLog' or 'None'

        @param params Parameters for the spiking nonlinearity, a list with [alpha, beta, theta]

        """

        self.nlType = nlType
        self.params = params

        if self.nlType == 'exp':
            self.nlFun = lambda g, params: params[0] * np.exp( params[1] * (g-params[2]))
        elif self.nlType == 'eLog':
            self.nlFun = lambda g, params: params[0] * np.log(1+ np.exp(params[1]*(g-params[2])))
        elif self.nlType== 'None':
            self.nlFun = lambda g, params: g

        self.estspknl = []

    def process(self, input):
        """
        :param input: generating signal
        :return: predicted firing rate
        """
        input = np.array(input)
        return self.nlFun(input, self.params)

    def histEstimate(self, g, gspk, binnum=20, display=0):

        bins = np.linspace(min(g), max(g), num=binnum)
        [gdist, bin_edges] = np.histogram(g, bins)
        [gdistspk, bin_edges] = np.histogram(gspk, bins)

        gdist = gdist.astype('float32')
        gdistspk = gdistspk.astype('float32')
        # bin centers
        NLx = (bin_edges[0:-1]+bin_edges[1:])*0.5
        NL = gdistspk/gdist;
        NL[gdist==0] = 0

        if display == 1:
            plt.plot(NLx, gdist/sum(gdist))
            plt.plot(NLx, gdistspk/sum(gdistspk))
            yl = plt.ylim()
            plt.plot(NLx, NL/max(NL)*yl[1])
            plt.show()

        estnl = {'NLx':NLx, 'NL':NL}
        return estnl

    def fitSpkNL(self, g, robs, display=0, hold_const = []):
        """
        :param g: Generating signal
        :param robs: Observed response (binned spike count)
        :param display: Bool value indicating whether to display NL
        :param hold_const: vector specifying which of the three parameters to hold constant [alpha beta theta]
        :return: None
        """
        g = np.array(g)
        init_params = copy.deepcopy(self.params)
        print " initial parameters: ", init_params

        # create constraints
        boundcontrs = [
                {'type' : 'ineq',
                 'fun'  : lambda x: np.array(x[0] - 1e-4),
                 'jac'  : lambda x: np.array([1.0, 0.0, 0.0])},
                {'type' : 'ineq',
                 'fun'  : lambda x: np.array(x[1] - 1e-4),
                 'jac'  : lambda x: np.array([0.0, 1.0, 0.0])},
                {'type' : 'ineq',
                 'fun'  : lambda x: np.array(x[2] + 1000),
                 'jac'  : lambda x: np.array([0.0, 1.0, 0.0])}]

        def Jac(i):
            j = np.zeros(3,dtype='float32')
            j[i] = 1
            return j

        eqconstr = []
        for i in range(len(hold_const)):
            def eqConstrFunc(x):
                print x, init_params, x[1]-init_params[i]
                return x[1]-init_params[i]

            if hold_const[i] > 0:
                eqconstr.append({   'type'  : 'eq',
                                    'fun'   : eqConstrFunc,
                                    'jac'   : lambda x: Jac(i)})

        cons = boundcontrs + eqconstr

        # todo: get constraint working here
        # print cons
        # res = minimize(spkNL.llSpkNL, init_params, args = (g, robs, self.nlType),\
        #                jac = spkNL.llSpkNL_deriv, constraints=cons, method = 'SLSQP', options = {'disp':True})

        res = minimize(llSpkNL, init_params, args = (g, robs, self.nlType), method='Nelder-Mead')

        self.params = res.x
        print " parameter after optimization: ", res.x

    def display(self):
        pass



