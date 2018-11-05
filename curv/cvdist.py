"""
Generator functions for various primitive random variables.
"""

from .cvrv import *

def distribution(name,chf,netInd=0):
	"""
	Return a new independent CRV defined by characteristic function CHF
	that has been added to Net netInd.

	Parameters:
		netInd (int): Index of netList Net that this RV is stored in
		chf (function): A one-argument function defining the CHF
	"""
	# Create the CRV
	X = CRV(name,[],netInd)
	# Read the current Net
	n = RV.netList[X.netInd]
	# Copy to immutable variables
	oldJoint = n.joint
	newNumArgs = n.numNodes # CRV has already updated numNodes
	# If this is not the first RV in the net:
	if newNumArgs != 1:
		def newJoint(newArgs):
			oldArgs = [newArgs[i] for i in range(newNumArgs) if i!=X.memInd]
			return chf(newArgs[X.memInd])*oldJoint(oldArgs)
		# Update the joint
		n.joint = newJoint
	else:
		def newFirstJoint(firstArg):
			return chf(firstArg[0])
		# Update the joint
		n.joint = newFirstJoint
	# Return the CRV
	return X

def normal(mu=0.,sigmasq=1.,netInd=0):
	"""
	Return a normally-distributed CRV of mean mu and variance sigmasq
	that has been added to Net netInd. 
	
	Parameters:
		netInd (int): Index of netList Net that this RV is stored in
		mu (float): Mean of the returned distribution
		sigmasq (float): Variance of the returned distribution
	"""
	name = 'N('+str(mu)+','+str(sigmasq)+')'
	chf = lambda t: np.exp(-0.5*sigmasq*(t**2) + 1j*mu*t)
	return distribution(name,chf,netInd)

def uniform(a=0.,b=1.,netInd=0):
	"""
	Return a uniform-distributed CRV from a to b that has been added to
	Net netInd. 
	
	Parameters:
		netInd (int): Index of netList Net that this RV is stored in
		a (float): Minimum value of the uniform
		b (float): Maximum value of the uniform
	"""
	name = 'U('+str(a)+','+str(b)+')'
	ib = 1j*b
	ia = 1j*a
	eps = np.spacing(1)
	chf = (lambda t: 
		(np.exp(ib*(eps+t)) - np.exp(ia*(eps+t)))/((eps+t)*(ib-ia)))
	return distribution(name,chf,netInd)
