import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

class model:
    
    def __init__(self, Ds, Rs, corr):
        
        '''
        'model' is a class describing unflolding model for two distributions
        extracted from the same data sample. Correlations between observed yields is
        therefore possible. The model is build from p sets of observed data with ni bins each,
        p respsonse matrices, and a correlation matrix between all observed yields.
        This model class allows derive stat-only unfolding, ie no nuisance parameter
        are included.
        
        Arguments
        ---------
         - Ds: list of 1D arrays of observed yields for the fist variable (n1, n2, ..., np)
         - Rs: lsit of 2D arrays of responses matrices ((n1, n1), (n2,n2), ... (np, np))
         - corr: 2D array of expected auto-correlation of all bins (n1+...+np, n1+...+np)
        '''
        
        # Array dimensions
        self.Ns = np.array([d.shape[0] for d in Ds])
        self.nPOIs = self.Ns.sum()
        
        # Storing all starting and ending inidice for internal use
        self._Nstart = [0] + list(self.Ns[:-1].cumsum())
        self._Nend   = list(self.Ns.cumsum())
        
        # Observed yields
        self.Ds = Ds
        
        # Response matrices
        self.Rs = Rs

        # Correlation between yields
        self.Corr = corr

        # Unfolded bins by matrix inversion
        self.Bs = [d @ np.linalg.inv(r) for d, r in zip(self.Ds, self.Rs) ]
    
                   
    def _array2list(self, x):
        '''
        Convert an 1D array of n1+...+np values into a list
        of p 1D arrays of shape n1, n2, ... np.
        '''
        return [x[n1:n2] for n1, n2 in zip(self._Nstart, self._Nend)]
                   
        
    def NLL(self, Bs):
    
        '''
        This function return the negative log likelihood of (d1, d2 | r1 x b1, r2 x b2)
        where data (d1, d2) are made of two observables looked at on 
        the same dataset: they are correlated. r1 and r2 are the response 
        matrix of each observable linking the truth yields b1 and b2, 
        which are the parameter of interest (POIs), to reco yields.

        Assuming only two bins per observables, for each observables separately, 
        likelihoods can be written as follow (we write b1, b2 = b, B and 
        d1, d1 = d, D temporarely):

            L(d | r x b) = P(d1 | r11 b1 + r12 b2) x P(d2 | r21 b1 + r22 b2)
            L(D | R x B) = P(D1 | R11 B1 + R12 B2) x P(D2 | R21 B1 + R22 B2)

        Each of these likelihood, under the assumption observables are gaussian,
        can be written using a multi-dimensionnal normal PDF.
            L(d | r, b) = Gauss(d, mu, Sigma)
            mu = (r11 b1 + r12 b2, r21 b1 + r22 b2)
            Sigma = diag(sqrt(m1), sqrt(mu2)) - no correlation between bins

        Looking at the same events on the two observables introduces some 
        correlations between d and D. One can use a n-dim normal PDF over
        d and D, with some non diagonal terms for Sigma.


        Arguments
        ---------
         - Bs: list of 1D array containing tested truth bin values for the 
               p observables. Shape: [n1, ..., np]

        Return
        ------
         - NNL, the negative log likelihood value.
        '''

        # Get array shape
        p = self.nPOIs

        # Vector of observations
        obs = np.concatenate(self.Ds)

        # Mean vector of the normal PDF: truth x response
        mu = np.concatenate([b @ r for (b, r) in zip(Bs, self.Rs)])

        # Correlation matrix of the normal PDF
        Sigma = np.zeros((p, p)) + 1e-12 * np.diag([1]*p)
        for i in range(p):
            for j in range(p):
                Sigma[i, j] = self.Corr[i, j] * np.sqrt(mu[i] * mu[j])

        # Compute the log likelihood
        delta  = obs - mu
        SigInv = np.linalg.inv(Sigma)
        NLL = delta.T @ SigInv @ delta

        # Return the result
        return NLL
    
    
    def minimizeNLL(self, iPOI=-1, vPOI=None):
        '''
        Minimize the NLL with possibly one POI kept constant.
        This POI is selected by its index iPOI and the value
        vPOI is set.
        '''
          
        # Function to minimize in case of full NLL
        def fullNLL(x):
            Bs = self._array2list(x)
            return self.NLL(Bs)
    
        # Function to minimize in case of one dixed POI
        def fixedNLL(x):
            b = np.zeros(self.nPOIs)
            b[:iPOI], b[iPOI], b[iPOI+1:] = x[:iPOI], vPOI, x[iPOI:]          
            Bs = self._array2list(b)
            return self.NLL(Bs) 
        
        # Initial starting point as matrix inverted truth bins
        x0 = np.concatenate(self.Bs)
        
        # Final function to minimize
        nll = fullNLL
        
        if iPOI > -1:
            # Protection
            if iPOI >= self.nPOIs:
                msg = 'POI index ({}) must be lower than N1+N1 ({})'
                raise NameError(msg.format(iPOI, self.nPOIs))
            
            # Change the function to minimize
            nll = fixedNLL
            
            # Change the initial values
            x0 = np.delete(x0, iPOI)           
        
        # Bounds
        xMax = np.max(x0) * 10
        xMin = np.min(x0) / 10
        bounds = [(xMin, xMax)] * x0.shape[0]
        
        # Minimization
        res = optimize.minimize(nll, x0=x0, tol=1e-6, method='Powell', bounds=bounds)
        
        return res
    
    
    def unfold(self):
        '''
        Return the a list of p 1D array of unfolded bins which minimize 
        the negative log likelihood of the problem, and the minum
        nll value:
       
        >>> m = model()
        >>> Bs, nllMin = m.unfold()
        
        The starting point of the minimization is given 
        the result of matrix inversion B0 = R^{-1} x D.
        '''
        
        # Run the full minimization
        res = self.minimizeNLL()
        
        # Return the unfolded bins
        return self._array2list(res.x), res.fun
    
    
    def profilePOI(self, iPOI, POImin, POImax, nScan=10):
        
        '''
        Perform a likelihood profiling for the parameter
        of interest indexed by iPOI, which must be strictly
        lower than n1+n2+...+np.
        
        An iterative approach is used to perform the profiling,
        based on the obtained NLL. Due to numerical instabilities,
        the postfit NLL might have unrelevant values. In that case,
        the value the POI is slightly shifted, and the fit is 
        performed again. The value is considered irrelevant if
          + NLL[i] > 1000, for the first POI value, ie i==0
          + NLL[i] > NLL[i-1] * 5
        Five iterations are allowed, when reached, the point is 
        disregarded.
        '''
        
        # Container for the results
        nlls = np.zeros(nScan)
        pois = np.zeros(nScan)
        dv   = (POImax -  POImin) / nScan
        
        # Loop over POI values
        for i, v in enumerate(np.linspace(POImin, POImax, nScan)):
            
            # Minimized at fixed
            res = self.minimizeNLL(iPOI=iPOI, vPOI=v)
            
            nllMin, iIter = res.fun, 0
            if i==0:
                criteria = lambda x: (x > 1000) and (iIter < 5)
            else:
                criteria = lambda x: (x > nlls[i-1]*5) and (iIter < 5)
            while criteria(nllMin):
                v += dv / 10.
                res = self.minimizeNLL(iPOI=iPOI, vPOI=v)
                nllMin = res.fun
                iIter += 1

            # Store the result
            if iIter < 4:
                pois[i] = v 
                nlls[i] = res.fun
            
        return pois, nlls
    
    
    def plotProfile(self, iPOI, POImin, POImax, nScan):    
        
        # Fit the NLL profile with a degree 3 polynom
        def f(x, a0, a1, a2, a3):
            return a0 + a1*x + a2*x**2 + a3*x**3
        
        # Run the profile and fit
        val, nll = self.profilePOI(iPOI, POImin, POImax, nScan)
        p, _ = optimize.curve_fit(f, val, nll)

        # Compute the chi2
        dy = f(val, *p) - nll
        chi2 = np.mean(dy**2)
        
        # Plot values and fit function
        plt.plot(val, nll, 'o', color='tab:blue')
        v = np.linspace(val.min(), val.max(), 1000)
        n = f(v, *p)
        plt.plot(v, n, '--', color='tab:blue', label='$\\chi^{2}=$' + '{:.2f}'.format(chi2))
        plt.legend()
        
    
    def postFitUncerPOIs(self, chi2Tol=1e-2):
        '''
        Return a list of p 2D array. Each of these array contains 
        the central values and uncertainties of all parameters of 
        interest, ie central value, negative uncertainty and 
        positive uncertainty for each unfolded bins. The list covers
        then the p unfoloded distribution. Final measurement is
            POI = nom -neg +pos
        
        >>> m = model(...)
        >>> poisHat = m.postFitUncerPOIs()
        >>> for distri in poisHat:
        >>>    for unfBin in distri:
        >>>       nom, neg, pos = unfBin
        
        An error is raised if the fit quality is not good, 
        which could be due to unstable LH minization, as treated
        in the function profilePOI().
        '''
        
        # Result container
        postFitPOIs = np.zeros((self.nPOIs, 3))
        
        # Determine best fit central value first
        Bs, nllMin = self.unfold()
        b = np.concatenate(Bs)
        
        # Loop over POIs
        for iPOI in range(self.nPOIs):
            
            # Get the central value of the current POI
            POI = b[iPOI]
            
            # First determine the proper range to scan
            # by having at least a dNLL>10
            dNLL, scale, v = 0, 0.05, POI
            while dNLL < 20:
                v = v * (1+scale)
                res = self.minimizeNLL(iPOI, v)
                dNLL = res.fun - nllMin
                
            # Profile the POI on the proper scale with 10 points
            dPOI = v-POI
            pMin, pMax, nScan = POI-dPOI, POI+dPOI, 10
            val, nll = self.profilePOI(iPOI, pMin, pMax, nScan)
            
            # Fit the NLL profile with a degree 3 polynom
            def f(x, a0, a1, a2, a3):
                return a0 + a1*x + a2*x**2 + a3*x**3
            
            # Fit the function
            p, _ = optimize.curve_fit(f, val, nll)
            
            # Get a continuous evolution
            v = np.linspace(val.min(), val.max(), 1000)
            n = f(v, *p)
            
            # Compute the chi2 by hand
            dy = f(val, *p) - nll
            chi2 = np.mean(dy**2)
            if chi2 > chi2Tol:
                msg  = 'WARNING: the chi2 fit is {:.2f}, investigate the profile LH for \n'
                msg += '--> (iPOI, Min, Max, N) = ({}, {},{}, {})'
                print(msg.format(chi2, iPOI,  pMin, pMax, nScan))
            
            iM = np.argmin(n)
            nM, nL, nR = n[iM], n[:iM], n[iM:]
            vM, vL, vR = v[iM], v[:iM], v[iM:]
            iL = np.argmin(np.abs(nL-(nM+1)))
            iR = np.argmin(np.abs(nR-(nM+1)))
    
            # Store the result for the current POI 
            postFitPOIs[iPOI] = np.array([vM, vM-vL[iL], vR[iR]-vM])
        
        # Return all the results
        return self._array2list(postFitPOIs)

