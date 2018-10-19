import numpy as np
import matplotlib.pyplot as plt

from .cvoperations import *
## PLOTTING METHODS

def pdplot(CRV,axLim=(None,None),Npoints=8192,tol=1e-2):
	# Plots the PDF of the given CRV

	# Compute the pdf using the CTFT on sampled points of the cfunc
	# Compute the sample boundaries using the chebyshev inequality
	tau = 2*np.pi
	k = tol**-0.5
	mu = E(CRV)
	si = V(CRV)**0.5
	start = mu - k*si/2
	span = k*si
	samplet = np.arange(-Npoints/(2*span), Npoints/(2*span), 1/span)
	vcfunc = np.vectorize(CRV.cfunc)
	samplecfun = vcfunc(tau*samplet)
	shift = np.exp(-1j*tau*samplet*start)
	pdf = Npoints/span*np.fft.ifft(np.fft.ifftshift(shift * samplecfun))
	ax = np.linspace(start,start + span - span/Npoints,Npoints)

	f, axes = plt.subplots()
	plt.plot(
		ax,np.real(pdf),
		ax,np.imag(pdf))
	plt.xlabel('x')
	plt.ylabel('pdf')
	plt.title('PDF of '+str(CRV))
	axes.set_xlim(axLim)
	plt.show()


def cfplot(CRV,argRange):
	ax = list(argRange)
	vals = list(map(CRV.cfunc,argRange))
	plt.plot(
		ax,[val.real for val in vals],
		ax,[val.imag for val in vals])
	plt.xlabel('t')
	plt.title('Characteristic function of '+str(CRV))
	plt.show()