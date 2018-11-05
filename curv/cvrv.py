"""
Classes and functions to create, hold, and operate on random variables (continuous and discrete) that exist on a Net.

Contents:

## Classes
class RV
class CRV(RV)
	method marginalCHF
	method moment

## Unary Operations
function E
function V

## Binary Operations
function addCrvNum
function addCrvCrv

"""

import numpy as np
from scipy.misc import derivative

from .cvnet import *

numTypes = (int, float, complex)


## Classes
class RV:
	"""
	Class of random variables that may be placed in a Net. 

	Hold all properties, methods, and operations of a random variable except for the CHF, which is held in the joint CHF of the Net containing the RV. Make sure that the joint chf is calculated before creating a new RV.

	Class attributes:
		netList (list): List of independent Nets an RV may be assigned to. By default there is one Net containing all RV instances.

	Properties:
		name (string): String representation of the RV
		varType (string): Sampling type ('continuous' or 'discrete')
		parents (list): RVs that influence this one (upstream of net)
		netInd (int): Index of netList Net that this RV is stored in
		memInd (int): Index of net.member list for this RV

	"""
	netList = [Net()] # List of Nets that hold the RV instances

	def __init__(self,name,varType,parents,netInd):
		"""Constructor for RV Class."""
		self.name = name # String representation of the RV
		self.varType = varType # Sampling type
		self.parents = parents # RVs that influence this one
		self.netInd = netInd # Index of netList Net
		self.memInd = RV.netList[netInd].newRV(self) # Index of net.member list for this RV

	def __repr__(self):
		"""Print the name of the RV."""
		return self.name

class CRV(RV):
	"""
	Class of continuous random variables.

	Methods:
		marginalCHF: Return the marginal CHF by slicing the Net.
		moment: Return the nth moment of the CRV.
	"""
	def __init__(self,name,parents=[],netInd=0):
		"""Constructor for CRV Class."""
		RV.__init__(self,name,'continuous',parents,netInd)

	def __add__(self,other):
		""" Addition of a CRV to a number or another CRV."""
		if isinstance(other,numTypes):
			return addCrvNum(self,other)
		elif isinstance(other,CRV):
			return addCrvCrv(self,other)
		else:
			return TypeError
	__radd__ = __add__
	def __neg__(self):
		""" Additive inverse of a CV."""
		return negCrv(self)
	def __sub__(self,other):
		""" Subtraction of a number or CV from a CV."""
		return self + (-other)
	def __rsub__(self,other):
		return -self + other
	def __mul__(self,other):
		return NotImplemented

	def marginalCHF(self):
		""" Return the marginal CHF of the RV by slicing the Net."""
		n = RV.netList[self.netInd]
		return lambda t: n.joint(
			[t if i==self.memInd else 0 for i in range(n.numNodes)])

	def moment(self,order=1,tol=1e-3):
		""" Return the nth moment of the CRV. """
		# Compute the nth derivative at t=0 to determine the nth moment
		# Set tolerance of derivative to 1e-3 to prevent rounding errors
		marginal = self.marginalCHF()
		return (-1j**order) * derivative(marginal,0,tol,order)

## UNARY OPERATIONS
def E(X):
	""" Return the expected value of an RV. """
	if isinstance(X,RV):
		return np.real(X.moment(1))
	else:
		return X

def V(X):
	""" Return the variance of an RV. """
	if isinstance(X,RV):
		return abs(X.moment(2) - X.moment(1)**2)
	else:
		return 0

## BINARY OPERATIONS
def addCrvNum(X,a):
	"""
	Add a CRV to a number and join result to the Net of the CRV.

	Parameters:
		X (CRV): CRV to be added
		a (int,float,complex): Number to be added

	Returns:
		CRV: A CRV representing the sum, stored in the same Net as X
	"""
	# Create the new RV
	Z = CRV(str(X)+'+'+str(a),[X],X.netInd)
	# Read the current net
	n = RV.netList[Z.netInd]
	# Copy
	oldJoint = n.joint
	# Use the formula derived in the README to compute new joint
	def newJoint(newArgs):
		oldArgs = [newArgs[X.memInd]+newArgs[Z.memInd] 
			if (i == X.memInd) else newArgs[i] 
			for i in range(n.numNodes-1)]
		return np.exp(1j*newArgs[Z.memInd]*a)*oldJoint(oldArgs)
	# Update the joint
	n.joint = newJoint
	# Return the CRV of the sum
	return Z

def addCrvCrv(X,Y):
	"""
	Add a CRV to a CRV and join result to the net of the CRVs.

	Parameters:
		X (CRV): First CRV to be added
		Y (CRV): Second CRV to be added

	Returns:
		CRV: A CRV representing the sum, stored in the same Net as X, Y
	"""

	# See docs for more information.
	# In implementation, we first create the new RV then update the joint chf of the net.

	# First make sure both RVs exist on the same net.
	assert (X.netInd == Y.netInd),\
		"During an addition, both RVs must belong to the same net."
	# Create the new RV
	Z = CRV(str(X)+'+'+str(Y), [X,Y], X.netInd)
	# Read the current Net
	n = RV.netList[Z.netInd]
	# Copy
	oldJoint = n.joint
	# Use the formula derived in the README to compute new joint
	def newJoint(newArgs):
		oldArgs = [
			newArgs[X.memInd]+newArgs[Z.memInd] 
			if (i==X.memInd) else
			newArgs[Y.memInd]+newArgs[Z.memInd]
			if (i==Y.memInd) else
			newArgs[i]
			for i in range(n.numNodes-1)]
		return oldJoint(oldArgs)
	# Update the joint
	n.joint = newJoint
	# Return the CRV of the sum
	return Z

