import numpy as np

from .cvrv import *

## OPERATIONS
def E(X):
	if isinstance(X,RV):
		return np.real(X.moment(1))
	else:
		return X

def V(X):
	if isinstance(X,RV):
		return abs(X.moment(2) - X.moment(1)**2)
	else:
		return 0