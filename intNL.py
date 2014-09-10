__author__ = 'ywcui'

import numpy as np
from matplotlib import pyplot as plt
from pylab import *
import copy

class intNL:
    " Class implementing the internal nonlinearity"

    def __init__(self,
                 type='nonparam',
                 NLx=[],
                 NL=[],
                 NLw=[]):
        """
        :param type: type of the nonlinearity, available options are
                    linear: f(x) = x
                    RecNL: f(x) = x if x>0 else 0
                    nonparm: nonparameteric representation with tent basis
        :param NLx: center of the tent basis
        :param NL: value for each tent basis
        :param NLw: width of each tent basis
        :return:
        """
        self.type = type
        self.prescale = 1.0

        if type is 'linear':
            return

        if NLx == []:
            NLx = np.linspace(-1.0, 1.0,21)
        else:
            NLx = np.array(NLx)
            NLx = NLx.astype('float32')

        if NL == []:
            NL = copy.deepcopy(NLx)
        else:
            NL = np.array(NL, dtype='float32')


        self.NLx = NLx
        self.NL = NL

        self.sigdistTick = 1.1*NLx
        self.sigdist = np.zeros_like(NLx)

        if NLw==[]:
            NLw = np.zeros_like(NLx)
            NLw[:-1] = NLx[1:]-NLx[:-1]
            NLw[-1] = NLw[-2]
        self.NLw = NLw

    def initialNL(self, NLxmin, NLxmax, NLtickN, type):
        NLx = np.linspace(NLxmin, NLxmax, NLtickN)
        if type is 'RecNL':
            self.NL = copy.deepcopy(NLx)
            self.NL[NLx<0.0] = 0.0
        elif type is 'linear':
            NL = copy.deepcopy(NLx)
        elif type is 'Quad':
            NL = np.power(NLx,2)

        self.NLx = NLx
        self.NL = NL
        self.sigdistTick = copy.deepcopy(self.NLx)
        self.sigdist = np.zeros_like(self.sigdistTick)

    def process(self, s):

        gs = self.prescale*s
        #gdist = np.histogram(input, self.sigdistTick)

        output = np.array(gs)

        if self.type == 'linear':
            return
        elif self.type == 'RecNL':
            output[gs<0] = 0.0
        elif self.type == 'Quad':
            output = np.power(output, 2)
        elif self.type == 'nonparam':
            output = intNL.nlin_proc_stim(gs, self.NL, self.NLx, self.NLw)

        output = output.reshape(gs.shape)

        return output

    def adjustScale(self, input, frac=95, eqbin=0, adjustNL=0):
        """
        adjust scale factor so that nonlinearity is operating at the appropriate range (adjust the X-axis)
        :param input: generating signal (numpy array)
        :param frac: fraction of the input signal that are covered by the nonlinearity
        :param eqbin: eq=1, adjust xtick s.t. there are equal number of samples in each bin
        :param adjustNL: adjust nonlinearity accordingly using either interpolation/extrapolation or
                            the parameteric functional form
        :return:
        """

        oldscale = copy.deepcopy(self.prescale)
        if frac<100:
            self.prescale = np.max(self.NLx[-1])/np.percentile(input,frac)
        else:
            self.prescale = np.max(self.NLx[-1])/np.max(np.abs(input))*100/frac

        nlticknum = len(self.NLx)
        tickstep = 100.0/(nlticknum-2)

        if eqbin:
            prctileTick = np.linspace(start=tickstep/2, stop=100-tickstep/2, num=nlticknum)
            self.NLx[1:-2] = np.percentile(input*self.prescale, prctileTick)
            self.sigdistTick = np.linspace(-1,1,21)*np.max(np.abs(self.NLx))

        self.sigdist = np.histogram(self.prescale*input, self.sigdistTick)

        #output = self.process(input)

        oldX = self.NLx/oldscale
        newX = self.NLx/self.prescale

        print "old scale is ", oldscale, " new scale is ", self.prescale

        if adjustNL:
            if self.type == 'normal':
                self.NL = np.interp(newX, oldX, self.NL)
            elif self.type == 'RecNL':
                self.NL = self.NLx * oldscale/self.prescale
                self.NL[self.NLx<0] = 0.0


    def makeXmatrix(self, s):
        """
        make x matrix for internal nonlinearity optimization
        :param s: generating matrix
        :return: x-matrix
        """
        NT = len(s)
        gs = self.prescale * s

        NN = len(self.NL)
        X = np.zeros((NT, NN), dtype='float32')

        for n in range(NN):
            if self.NLw != []:
                X[:,n] = X[:,n] + intNL.piece_proc_stim(gs, self.NLx[n], [self.NLx[n]-self.NLw[n], self.NLx[n]+self.NLw[n]])
            else:
                if n==0:
                    X[:,n]=X[:,n]+intNL.piece_proc_stim(gs, self.NLx[n], self.NLx[n+1])
                elif n==NN-1:
                    X[:,n]=X[:,n]+intNL.piece_proc_stim(gs, self.NLx[n], self.NLx[n-1])
                else:
                    X[:,n]=X[:,n]+intNL.piece_proc_stim(gs, self.NLx[n], [self.NLx[n-1],self.NLx[n+1]])

        k = self.NL

        return X, k

    def setK(self, K, offset=0):
        self.NL = K[offset+np.arange(len(self.NL))]
        offset += len(self.NL)
        return offset

    def NLderivative(self):
        """
        calculate derivative of nonlinearity
        :return:
        """

        fpr = np.zeros(len(self.NLx)-1,dtype='float32')
        for i in range(len(fpr)):
            fpr[i] = self.prescale*(self.NL[i+1]-self.NL[i])/(self.NLx[i+1]-self.NLx[i])

        self.dNL = fpr

    def display(self):
        if self.NLx == []:
            self.NLx = np.linspace(-1,1,21)

        # define x-axis
        if self.sigdistX != []:
            xticks = np.linspace(-1,1,41)*np.max(np.abs(self.sigdistX))
        else:
            xticks = self.NLx

        if self.prescale!=0:
            NLfunc = self.process(xticks/self.prescale)
        else:
            NLfunc = np.zeros_like(xticks)

        plt.plot(xticks, NLfunc)
        plt.plot(self.NLx, self.NL)
        yl = plt.ylim()

        if self.sigdist!=[] and self.sigdistTick!=[]:
            plt.plot(self.sigdistX, self.sigdist/np.max(self.sigdist)*(yl[1]-yl[0])+yl[0],'r-');

        plt.xlim(self.NLx[0], self.NLx[-1])
        axis('equal')

    @staticmethod
    def nlin_proc_stim(stim, NL, NLx, NLw=[]):
        """
        Process stimulus with a nonlinear function which is expressed
        with tent basis
        :param stim:
        :param NL:
        :param NLx:
        :param NLw:
        :return: f[stim]
        """
        NT = stim.shape[0]
        if len(stim.shape) == 1:
            Nx = 1
        elif len(stim.shape) == 2:
            Nx = stim.shape[1]

        stim = stim.reshape((NT*Nx))
        sdist = np.zeros_like(NL)

        # cutoff
        # stim[stim<NLx[0]] = NL[0]
        # stim[stim>NLx[-1]] = NL[-1]

        if NLw == []:
            NLxAug = np.append(NLx, NLx[-1] + 1)
            NLAug = np.append(NL, 0)
            # Return the indices of the bins to which each value in stim belongs
            binindx = np.digitize(stim, NLxAug)
            # for each value in the stim array, get the location of the left/right tent
            ledge = NLxAug[binindx-1]
            redge = NLxAug[binindx]

            binwidth = redge - ledge
            # distance to the left/right tent
            d2ledge = stim - ledge
            d2redge = redge - stim

            # contribution from the left/right tent
            lcontr = 1-d2ledge/binwidth
            rcontr = 1-d2redge/binwidth

            lcoeff = NLAug[binindx]
            rcoeff = NLAug[binindx]

            processedStim = lcontr*lcoeff + rcontr*rcoeff
        else:
            processedStim = np.zeros_like(stim)
            for j in range(len(NLx)):
                # calculate output of the jth basis
                basisj = intNL.piece_proc_stim( stim, NLx[j], [NLx[j]-NLw[j], NLx[j]+NLw[j]])
                processedStim = processedStim + NL[j] * basisj
                sdist[j] = sum(basisj!=0);

        processedStim = processedStim.reshape(stim.shape)

        return processedStim

    @staticmethod
    def piece_proc_stim(s, center, boundpts):
        """
        Process stimulus with piecewise linear tent basis t[.]
        :param s: an array of stimulus to be processed
        :param center: center of the tent
        :param boundpts: boundary points of tent basis
        :return: t[stim]
        """
        # rl: distance to left edge
        # rr: distance to right edge
        if len(boundpts) == 2:
            rl = center-boundpts[0]
            rr = boundpts[1]-center
        elif len(boundpts) == 1:
            if boundpts < center:
                rl = center - boundpts
                rr = None
            else:
                rl = None
                rr = boundpts - center

        proc_s = np.zeros_like(s)
        # Left Side of Tent
        if rl != None:
            indx = np.logical_and(s>center-rl, s<=center)
            proc_s[indx] = 1 - (center-s[indx])/rl
        else:
            proc_s[s <= center] = 1.0

        # Right Side of Tent
        if rr != None:
            indx = np.logical_and(s>center, s<=center+rr)
            proc_s[indx] = 1 - (s[indx]-center)/rr
        else:
            proc_s[s > center] = 1.0

        return proc_s

    @staticmethod
    def fprime(x, fpr, NLx):
        """
        y = f'(x)
        :param x:
        :param fpr:
        :param NLx:
        :return:
        """
        y = np.zeros_like(x, dtype='float32')
        for i in range(len(NLx)-1):
            y[np.logical_and(x >= NLx[i], x<NLx[i+1])] = fpr[i]

        y[x > NLx[-1]] = 0.0
        y[x < NLx[0] ] = 0.0