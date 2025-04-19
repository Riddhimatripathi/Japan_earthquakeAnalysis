import numpy as np
from sklearn.feature_selection import mutual_info_regression
from pyswarm import pso

def select_features_pso(X, y, mi_scores, seed=42):
    def obj(w):
        sel = w > 0.5
        if not np.any(sel):
            return 1e6
        return -np.mean(mi_scores[sel])
    lb = [0]*X.shape[1]
    ub = [1]*X.shape[1]
    weights,_ = pso(obj, lb, ub, swarmsize=10, maxiter=10)
    return X.columns[weights>0.5]
