import numpy as np
from scipy.misc import derivative

numTypes = (int,float,complex)

class RV:
	def __init__(self,vartype,name):
		self.vartype = vartype # 'continuous' or 'discrete'
		self.name = name # Name of the variable

class CRV(RV):
	def __init__(self,cfunc,dim,name):
		RV.__init__(self,'continuous',name)
		self.cfunc = cfunc # Characteristic function
		self.dim = dim # Dimensionality
		self.name = name # Name

	def __repr__(self):
		return self.name # Print the name

	## ARITHMETIC METHODS

	def __add__(self,other):
		if isinstance(other,numTypes):
			return CRV(
				lambda t: self.cfunc(t)*np.exp(1j*t*other),
				self.dim,
				str(self)+'+'+str(other))
		if isinstance(other,CRV):
			return CRV(
				lambda t: self.cfunc(t)*other.cfunc(t),
				self.dim,
				str(self)+'+'+str(other))
	__radd__ = __add__

	def __neg__(self):
		return CRV(
			lambda t: self.cfunc(-t),
			self.dim,
			'-'+str(self))
	def __sub__(self,other):
		return self + (-other)
	def __rsub__(self,other):
		return -self + other

	def __mul__(self,other):
		if isinstance(other,numTypes):
			return CRV(
				lambda t: self.cfunc(t*other), # Something wrong here
				self.dim,str(other)+'*'+str(self))
		if isinstance(other,CRV):
			return NotImplemented
	__rmul__ = __mul__

	## RANDOM VARIABLE METHODS

	def moment(self,order=1,tol=1e-3):
		# Returns the nth moment of the CRV

		# Compute the nth derivative at t=0 to determine the nth moment
		# Set tolerance of derivative to 1e-3 to prevent rounding errors
		return (-1j**order) * derivative(self.cfunc,0,tol,order)