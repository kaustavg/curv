import numpy as np

from .cvrv import *

## COMMON DISTRIBUTIONS
def Normal(mu=0, sigmasq=1):
	return CRV(
		lambda t: np.exp(-0.5*sigmasq*(t**2) + 1j*mu*t),
		1,'N('+str(mu)+','+str(sigmasq)+')')

def Uniform(a=0, b=1):
	ib = 1j*b
	ia = 1j*a
	eps = np.spacing(1)
	return CRV(
		lambda t: (np.exp(ib*(eps+t)) - np.exp(ia*(eps+t)))/((eps+t)*(ib-ia)),
		1,'U('+str(a)+','+str(b)+')')