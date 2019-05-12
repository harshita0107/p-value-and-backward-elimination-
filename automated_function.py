# -*- coding: utf-8 -*-
"""
Created on Sun May 12 14:23:14 2019

@author: Harshita
"""
import statsmodels.formula.api as sm
def backwardElimination(x,sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars-i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
        regressor_OLS.summary()
        return x
SL = 0.05
X_opt = X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]]
X_Modeled = backwardElimination(X_opt, SL)

