import numpy as np

from .cvrv import *

class Net:
	def __init__(self):
		self.numNodes = 0
		self.members = []
		self.dag = dict()
		self.joint = lambda: None

	def joinIndepRV(self,rv):
		# Adds a new independent RV to the joint cfun
		self.numNodes += 1
		self.members.append(rv)
		self.dag[rv] = []
		if self.numNodes != 1:
			self.joint = lambda arg: rv.cfunc(arg[-1])*self.joint(arg[:-1])
		else:
			self.joint = rv.cfunc

	def getRVIndex(self,rv):
		return n.members.index(rv)

	## COMMON DISTRIBUTIONS
	def normal(self,mu=0, sigmasq=1):
		crv = CRV(
			lambda t: np.exp(-0.5*sigmasq*(t**2) + 1j*mu*t),
			1,
			'N('+str(mu)+','+str(sigmasq)+')',
			self)
		self.joinIndepRV(crv)
		return crv

	def uniform(self,a=0, b=1):
		ib = 1j*b
		ia = 1j*a
		eps = np.spacing(1)
		crv = CRV(
			lambda t: (np.exp(ib*(eps+t)) - np.exp(ia*(eps+t)))/((eps+t)*(ib-ia)),
			1,
			'U('+str(a)+','+str(b)+')',
			self)
		self.joinIndepRV(crv)
		return crv