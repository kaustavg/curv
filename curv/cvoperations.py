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

def cvcvadd(X,Y):
	if not isinstance(X,RV) or not isinstance(Y,RV):
		return TypeError

	n = X.net
	Xind = n.getRVIndex(X)
	Yind = n.getRVIndex(Y)
	# To compute the new joint pdf, first compute the result of the
	# projection via central slice theorem
	# Note: all other arguments are 0
	proj = lambda t: n.joint(
		[t if (i in [Xind, Yind]) else 0 for i in range(n.numNodes)])
	# Add the projection to the joint net
	def jointfun(arg):
		if (arg[Xind] == arg[Yind]) and (arg[-1] == arg[Xind]):
			return n.joint(arg[:-1])*proj(arg[-1])
		else:
			return 0
	n.joint = jointfun