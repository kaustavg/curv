import numpy as np
from scipy.misc import derivative

numTypes = (int,float,complex)

class RV:
	def __init__(self,vartype,name,net):
		self.vartype = vartype # 'continuous' or 'discrete'
		self.name = name # Name of the variable
		self.net = net # Ref to net the RV belongs to
		
class CRV(RV):
	def __init__(self,cfunc,dim,name,net):
		RV.__init__(self,'continuous',name,net)
		self.cfunc = cfunc # Characteristic function
		self.dim = dim # Dimensionality

	def __repr__(self):
		return self.name # Print the name

	## ARITHMETIC METHODS

	def __add__(self,other):
		if isinstance(other,numTypes):
			return CRV(
				lambda t: self.cfunc(t)*np.exp(1j*t*other),
				self.dim,
				str(self)+'+'+str(other),
				self.net)
		if isinstance(other,CRV):
			return CRV(
				lambda t: self.cfunc(t)*other.cfunc(t),
				self.dim,
				str(self)+'+'+str(other),
				self.net)
	__radd__ = __add__

	def __neg__(self):
		return CRV(
			lambda t: self.cfunc(-t),
			self.dim,
			'-'+str(self),
			self.net)
	def __sub__(self,other):
		return self + (-other)
	def __rsub__(self,other):
		return -self + other

	def __mul__(self,other):
		if isinstance(other,numTypes):
			return CRV(
				lambda t: self.cfunc(t*other), # Something wrong here
				self.dim,
				str(other)+'*'+str(self),
				self.net)
		if isinstance(other,CRV):
			return NotImplemented
	__rmul__ = __mul__

	## RANDOM VARIABLE METHODS

	def moment(self,order=1,tol=1e-3):
		# Returns the nth moment of the CRV

		# Compute the nth derivative at t=0 to determine the nth moment
		# Set tolerance of derivative to 1e-3 to prevent rounding errors
		return (-1j**order) * derivative(self.cfunc,0,tol,order)