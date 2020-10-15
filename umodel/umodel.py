import numpy as np
from scipy import optimize

class model:
    
    def __init__(self, d1, d2, r1, r2, corr):
        
        '''
        'model' is a class describing unflolding model for two distributions
        extracted from the same data sample. Correlations between observed yields is
        therefore possible. The model is build from two sets of observed data,
        two respsonse matrices, and a correlation matrix between observed yields.
        This model class allows derive stat-only unfolding, ie no nuisance parameter
        are included.
        
        Arguments
        ---------
         - d1: 1D array of observed yields for the fist variable (n)
         - d2: 1D array of observed yields for the second variable (N)
         - r1: 2D array of response matrix for the first variable (n, n)
         - r2: 2D array of response matrix for the second variable (N, N)
         - corr: 1D array of expected auto-correlation of all bins (n+N, n+N)
        '''
        
        # Array dimensions
        self.N1, self.N2 = d1.shape[0], d2.shape[0]
        self.nPOIs = self.N1 + self.N2 
        
        # Observed yields
        self.data1, self.data2 = d1, d2
        
        # Response matrices
        self.Resp1, self.Resp2 = r1, r2

        # Correlation between yields
        self.Corr = corr

        # Unfolded bins by matrix inversion
        self.b1 = self.data1 @ np.linalg.inv(self.Resp1)
        self.b2 = self.data2 @ np.linalg.inv(self.Resp2)
        
        
    def NLL(self, b1, b2):
    
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
         - b1: 1D array of truth yields for the first variable (n)
         - b2: 1D array of truth yields for the second variable (N)

        Return
        ------
         - NNL, the negative log likelihood value.
        '''

        # Get array shape
        p = self.nPOIs

        # Vector of observations
        obs = np.concatenate([self.data1, self.data2])

        # Mean vector of the normal PDF
        mu1 = b1 @ self.Resp1
        mu2 = b2 @ self.Resp2
        mu = np.concatenate([mu1, mu2])

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
    
    
    def minimizeNLL(self, iPOI=-1, vPOI=None, b1start=np.array([]), b2start=np.array([])):
        '''
        Minimize the NLL with possibly one POI kept constant.
        This POI is selected by its index iPOI and the value
        vPOI is set.
        '''
                
        # Function to minimize in case of full NLL
        def fullNLL(x):
            b1 = x[:self.N1]
            b2 = x[self.N1:]
            return self.NLL(b1, b2)
    
        # Function to minimize in case of one dixed POI
        def fixedNLL(x):
            b = np.zeros(self.nPOIs)
            b[:iPOI], b[iPOI], b[iPOI+1:] = x[:iPOI], vPOI, x[iPOI:]          
            b1, b2 = b[:self.N1], b[self.N1:]
            return self.NLL(b1, b2) 
        
        # Initial starting point
        if b1start.size == 0:
            b1start = self.b1
        if b2start.size == 0:
            b2start = self.b2
        x0 = np.concatenate([b1start, b2start])
        
        # Final function to minimize
        nll = fullNLL
        
        if iPOI > -1:
            # Protection
            if iPOI >= self.N1+self.N2:
                msg = 'POI index ({}) must be lower than N1+N1 ({})'
                raise NameError(msg.format(iPOI, self.nPOIs))
            
            # Change the function to minimize
            nll = fixedNLL
            
            # Change the initial values
            x0 = np.delete(x0, iPOI)           
        
        # Bounds
        xMax = np.max(x0) * 100
        xMin = np.min(x0) / 100
        bounds = [(xMin, xMax)] * x0.shape[0]
        
        # Minimization
        res = optimize.minimize(nll, x0=x0, tol=1e-6, method='Powell', bounds=bounds)
        
        return res
    
    
    def unfold(self, b1start=np.array([]), b2start=np.array([])):
        '''
        Return the two sets of unfolded bins which minimize 
        the negative log likelihood of the problem, and the minum
        nll value:
       
        >>> m = model()
        >>> b1, b2, nllMin = m.unfold()
        
        b1start and b2start are the starting point of the 
        minimization. Default is 'None', meaning that the 
        starting point is given the result of matrix 
        inversion B0 = R^{-1} x D.
        '''
        
        # Run the full minimization
        res = self.minimizeNLL(b1start=b1start, b2start=b2start)
        
        # Return the unfolded bins
        return res.x[:self.N1], res.x[self.N1:], res.fun
    
    
    def profilePOI(self, iPOI, POImin, POImax, nScan=10):
        
        '''
        Perform a likelihood profiling for the parameter
        of interest indexed by iPOI, which must be strictly
        lower than N1+N2.
        '''
        
        # Container for the results
        nlls = np.zeros(nScan)
        pois = np.linspace(POImin, POImax, nScan)
        
        # Loop over POI values
        for i, v in enumerate(pois):
            
            # Minimized at fixed
            res = self.minimizeNLL(iPOI=iPOI, vPOI=v)
            
            # Store the result
            nlls[i] = res.fun
            
        return pois, nlls
    
    
    def postFitUncerPOIs(self):
        '''
        Return central values and uncertainties of all parameters of 
        interest, ie central value, negative uncertainty and 
        positive uncertainty for each. Final measurement is then
            POI = nom -neg +pos
        
        >>> m = model(...)
        >>> poisHat = m.postFitUncerPOIs()
        >>> for poi in poisHat:
        >>>    nom, neg, pos = poi
        '''
        
        # Result container
        postFitPOIs = np.zeros((self.nPOIs, 3))
        
        # Determine best fit central value first
        b1, b2, nllMin = self.unfold()
        b = np.concatenate([b1, b2])
        
        # Loop over POIs
        for iPOI in range(self.nPOIs):
            
            # Get the central value of the current POI
            POI = b[iPOI]
            
            # First determine the proper range to scan
            # by having at least a dNLL>10
            dNLL, scale = 0, 0.2
            while dNLL < 10:
                v = POI * (1+scale)
                res = self.minimizeNLL(iPOI, v)
                dNLL = res.fun - nllMin
                scale += 0.1
                
            # Profile the POI on the proper scale with 10 points
            pMin, pMax, nScan = POI*(1-scale), POI*(1+scale), 10
            val, nll = self.profilePOI(iPOI, pMin, pMax, nScan)
            
            # Fit the NLL profile with a degree 3 polynom
            def f(x, a0, a1, a2, a3):
                return a0 + a1*x + a2*x**2 + a3*x**3
            
            # Fit the function
            p, _ = optimize.curve_fit(f, val, nll)
            
            # Get a continuous evolution
            v = np.linspace(val.min(), val.max(), 1000)
            n = f(v, *p)
            
            iM = np.argmin(n)
            nM, nL, nR = n[iM], n[:iM], n[iM:]
            vM, vL, vR = v[iM], v[:iM], v[iM:]
            iL = np.argmin(np.abs(nL-(nM+1)))
            iR = np.argmin(np.abs(nR-(nM+1)))
    
            # Store the result for the current POI 
            postFitPOIs[iPOI] = np.array([vM, vM-vL[iL], vR[iR]-vM])
        
        # Return all the results
        return postFitPOIs

