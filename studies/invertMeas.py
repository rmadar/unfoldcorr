import numpy as np
import matplotlib.pyplot as plt
import json
from scipy import optimize
import umodel as unf

import matplotlib as mpl
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 'xx-large'
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['lines.linewidth'] = 2.5
mpl.rcParams['figure.figsize'] = (10, 7)


# Loading data cos(n, -)
with open('../data/CosThetaNminus/asimov_spinCorrelation.json', 'r') as read_file:
    dm = np.array(json.load(read_file))
    
with open('../data/CosThetaNminus/resmat_spinCorrelation.json', 'r') as read_file:
    rm = np.array(json.load(read_file))

    
# Loading data cos(n, +)
with open('../data/CosThetaNplus/asimov_spinCorrelation.json', 'r') as read_file:
    dp = np.array(json.load(read_file))
    
with open('../data/CosThetaNplus/resmat_spinCorrelation.json', 'r') as read_file:
    rp = np.array(json.load(read_file))

# Loading data correlations between the two observables
with open('../data/correlation_matrix_CosThetaNplus_CosThetaNminus_bins.json', 'r') as read_file:
    corrDict = json.load(read_file)

n, p = rm.shape[0], rp.shape[0]
corr = np.zeros((n+p, n+p))
for i, (iName, line) in enumerate(corrDict.items()):
    for j, (jName, c) in enumerate(line.items()):
        corr[i, j] = c


# Get the first unfolding models
Ds = [dp, dm]
Rs = [rp, rm]
m1Corr = unf.model(Ds, Rs, corr)
m1     = unf.model(Ds, Rs, corr=np.diag([1]*8))       


# Now to the same thing but inversing the two observables
D2s = [dm, dp]
R2s = [rm, rp]
corr2 = corr.copy()
for i in range(4):
    for j in range(4):  
        corr2[i, j] = corr[i+4, j+4]
        corr2[i+4, j+4] = corr[i, j]
        corr2[i, j+4] = corr[i+4, j]
        corr2[i+4, j] = corr[i, j+4]
m2Corr = unf.model(D2s, R2s, corr2, backend='minuit')
m2     = unf.model(D2s, R2s, corr=np.diag([1]*8), backend='minuit')

# Print correlation matrix
np.set_printoptions(precision=2)
print('[+, -]\n', corr , '\n')
print('[-, +]\n', corr2, '\n')


# Perform and compare thet two unfolding
res2, nll2 = m2Corr.unfold()
res1, nll1 = m1Corr.unfold()
print('\nCompare best-fit values cos(+):')
print(res1[0])
print(res2[1])
print('\nCompare best-fit values cos(-):')
print(res1[1])
print(res2[0])



print('\nCompare NLL evaluation on the same point:')
Bm = np.array([ 1.51049192e+10,  1.51069154e+10,  1.51071235e+10,  1.51046997e+10])
Bp = np.array([-6.36546195e+08, -6.33778546e+08, -6.33710702e+08, -6.36499063e+08])
print('NLL[-, +](Bm, Bp) =', m2Corr.NLL([Bm, Bp]))
print('NLL[+, -](Bp, Bm) =', m1Corr.NLL([Bp, Bm]))
print('\n')
