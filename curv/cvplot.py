"""
Functions for the plotting and display of Nets and RVs.
"""
# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt

from .cvrv import *

def plotPD(X,axLim=(None,None),Npoints=8192,tol=1e-2):
	""" 
	Plot the PDF of the given RV.
	
	Parameters:
		axLim (tuple): x axis limits to plot (xmin, xmax)
		Npoints (int): Number of points to sample
		tol (float): Tolerance of PDF computation
	"""

	# Compute the pdf using the CTFT on sampled points of the chf
	# Compute the sample boundaries using the chebyshev inequality
	tau = 2*np.pi
	k = tol**-0.5
	mu = E(X)
	si = V(X)**0.5
	assert (si!=0),"Variance of RV cannot be 0. Improper distribution."
	start = mu - k*si/2
	span = k*si
	samplet = np.arange(-Npoints/(2*span), Npoints/(2*span), 1/span)
	vchf = np.vectorize(X.marginalCHF())
	samplechf = vchf(tau*samplet)
	shift = np.exp(-1j*tau*samplet*start)
	pdf = Npoints/span*np.fft.ifft(np.fft.ifftshift(shift * samplechf))
	ax = np.linspace(start,start + span - span/Npoints,Npoints)

	f, axes = plt.subplots()
	plt.plot(
		ax,np.real(pdf),
		ax,np.imag(pdf))
	plt.xlabel('x')
	plt.ylabel('pdf')
	plt.title('PDF of '+str(X))
	axes.set_xlim(axLim)
	plt.show()

def plotCHF(X,argRange,Npoints=8192):
	""" 
	Plot the complex CHF of the given RV.

	Parameters:
		argRange (list): Axes limits as [xmin, xmax, ymin, ymax]
		Npoints (int): Number of points to sample
	"""

	ax = np.linspace(argRange[0],argRange[1],Npoints)
	vals = np.array(list(map(X.marginalCHF(),ax)))
	print(vals)
	plt.plot(
		ax,[val.real for val in vals],
		ax,[val.imag for val in vals])
	plt.xlabel('t')
	plt.title('Characteristic function of '+str(X))
	plt.show()

def plotNet(netInd=0):
	"""
	Print a topologically ordered graph of the Net in index netInd.
	
	Parameters:
		netInd (int): Index of net in RV.netList to be queried 
	"""
	# UTF-8 symbols for plotting DAG
	pipe = '│'
	dash = '─'
	cor = '╰'
	dia = '╲'
	branch = '├'
	merge = '┴' # May also use cor
	cross = branch # May also use '┼'
	arrow = dash # May also use '╼' or '►'

	# Grab the Net
	n = RV.netList[netInd]
	# Plot the DAG in topological order with appropriate edges
	topSorted = n.topSort()
	num = len(topSorted)
	# Create canvas
	canvas = [[' ' for i in range(j+1)] for j in range(num*2-1)]
	# Add labels
	for i in range(num):
		canvas[i*2] += str(topSorted[i])
	# Add diagonal edges
	for i in range(num-1):
		if topSorted[i+1] in n.dag[topSorted[i]]:
			canvas[2*i+1] += dia

	def drawEdges(start,end):
		'''Helper for drawing cornered edges.'''
		if end-start == 2: # Do diagonal instead
			return
		start += 1 # For alignment
		for row in range(start,end): # Vertical bars
			if canvas[row][start] == cor:
				canvas[row][start] = branch
			elif canvas[row][start] == merge:
				canvas[row][start] = cross
			else:
				canvas[row][start] = pipe
		if canvas[end][start] == dash: # Corners
			canvas[end][start] = merge
		else:
			canvas[end][start] = cor
		for col in range(start+1,end): # Horizontal bars
			canvas[end][col] = dash
		canvas[end][end] = arrow


	# Draw the edges
	for i in range(num):
		children = n.dag[topSorted[i]]
		children.sort(key=lambda c:topSorted.index(c))
		start = i*2
		for j in range(len(children)):
			end = 2*topSorted.index(children[j])
			drawEdges(start,end)

	# Print
	for i in range(len(canvas)):
		print(''.join(canvas[i]))