import numpy as np
import umodel as unf

nBins = 1
Ds = [np.array([100]*nBins), np.array([100]*nBins)]
Rs = [np.eye(nBins), np.eye(nBins)]

def corrMatrix(rho):
    unity = np.eye(2*nBins)
    fullCorr = unity.copy()
    for i in range(0, nBins):
        for j in range(0, nBins):
            if i == j:
                fullCorr[i+nBins, j] = rho
                fullCorr[i, j+nBins] = rho
    return fullCorr


print('\nMinimum NLL as function of the correlation:')
for r in np.linspace(-1, 1, 10):
    m = unf.model(Ds, Rs, corrMatrix(r))
    _, nll = m.unfold()
    print('{:.2f}: {:.4e}'.format(r, nll))

print('\nNLL(b1, b2) as function of the correlation:')
for r in np.linspace(-1, 1, 10):
    m = unf.model(Ds, Rs, corrMatrix(r))
    nll = m.NLL([[98], [123]])
    print('{:.2f}: {:.4e}'.format(r, nll))
    
