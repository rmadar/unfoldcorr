# Unfolding with correlated observables

This repository holds a python code to perform unfolding (without systematic uncertainties) based on
correlated observables. The typical use case is to unfold simultaneoulsy two distributions of the
same dataset, where bins would be then correlated.

## In a nutshell

```python
import umodel as unf

# Observed yields
d1, d2 = np.array([ 90, 110]),  np.array([170,  30])
Ds = [d1, d2]

# Response matrices
r1, r2 = np.diag([1, 1]), np.diag([1, 1])
Rs = [r1, r2]

# Correlation matrix
corr = np.diag([1, 1, 1, 1])

# Build the unfolding model
m = unf.model(Ds, Rs, corr)

# Get central values (NLL mimimization)
cVals, minNLL = m.unfold()

# Run a profile NLL for the truth bin i
vPOI, vNLL = m.profilePOI(i, xMin, xMax, nScan)

# Get Post-fit POIs with profiledNLL-based uncertainties
postFitPOIs = m.postFitUncerPOIs()
```