"""
Classes to work with collections of RVs in a Bayesian network.
"""

class Net:
	"""
	Container for multiple RVs in a Bayesian Network.

	Properties:
		numNodes (int): Total number of RVs in the net
		members (list): List of RVs in the net
		dag (dict): DAG relating the dependencies of the RVs in the net
		joint (function): Joint CHF of RVs in the net
			Parameters:
				args: list of numNodes inputs to RVs in joint chf in same order as members
			Returns:
				float: joint chf at specified value of args

	Methods:
		newRV: Assigns a new RV to the net and returns its memInd.
		topSort: Return a topologically sorted member list of the net.
	"""

	def __init__(self):
		"""Constructor for Net class."""
		self.numNodes = 0 # Total number of RVs in net
		self.members = [] # List of RVs in net
		self.dag = dict() # Unlabeled DAG containing the Bayesian net
		self.joint = lambda args: None # Joint chf of all the RVs

	def newRV(self,X):
		""" 
		Assign a new RV to the Net and return its memInd. Do not update the net joint chf. This method is only called by the RV constructor.
		
		Parameters:
			x (RV): RV to be added to the Net
		Returns:
			int: memInd of the RV in the Net
		"""

		self.numNodes += 1
		self.members.append(X)
		self.dag[X] = []
		for parent in X.parents:
			self.dag[parent].append(X)
		return self.numNodes-1

	def topSort(self):
		"""Return a topologically sorted member list of the net."""
		graph = self.dag
		seen = set()
		stack = []
		order = []
		q = self.members.copy()
		while q:
			v = q.pop()
			if v not in seen:
				seen.add(v)
				q.extend(graph[v])
				while stack and v not in graph[stack[-1]]:
					order.append(stack.pop())
				stack.append(v)
		return stack + order[::-1]