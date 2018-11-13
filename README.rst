====
curv
====

**curv (Continuous Random Variables)** is a python package for creating 
and manipulating continuous random variables in a bayesian network. You 
may find it useful for performing bayesian inference on "broad" 
distributions with  high kurtosis or highly dimensional joint 
distributions where discrete sampling would require excessive memory.

Typical usage often looks like this::
	
    #!/usr/bin/env python

    import curv as cv

    A = cv.normal(1,2)
    B = cv.uniform(-5,4)
    C = 4 + A - 2*B
    # D = C * A**2 # Not implemented
    cv.plotPDF(D,-10,10)
    cv.plotNet()
    cv.plotCHF(D,-10,10)

Installation & Dependencies
===========================

Dependencies include:

* Numpy

* Scipy

* Matplotlib

Usage
=====
Curv stores its random variables as continuous functions as opposed to storing discrete probabilities in sampled bins. This data
structure reduces memory consumption (since distributions are not 
sampled and stored as individual points). 

As a result, curv would be useful for handling large, highly dependent bayesian networks with high dimensional joint probability distribution functions without second-order approxiations (such as Chow-Liu trees).

Curv would also be useful for distributions with high kurtosis, where discrete sampling over a large number of bins would be needed to capture rare events at the edges of the distribution.

Creating a Bayesian Network
---------------------------

Consider the following: A and B are both primitive RVs. Then define:
C = A + B
Since A and B are primitives and therefore independent, C's pdf is the convolution of A and B's pdfs. Now consider the following:
D = A + C
This is not a straightforward convolution of the pdfs, since C is not  independent with A. Instead, this is a projection of the joint pdf  perpendicular to the y=x line. Since we are internally storing the  characteristic functions, this projection is equivalent to slicing the joint cfun along the line y=x and scaling the result (by central-slice theorm).

Performing Inference
--------------------

Displaying Results
------------------

Under the Hood
==============

Characteristic Functions
------------------------

Joint Probability Characteristic Functions
------------------------------------------

Arithmetic Operations on Joint Characteristic Functions
-------------------------------------------------------

Addition of a Constant to a Random Variable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Our problem is to find the joint CHF :math:`\phi_{X,Y}(t_x,t_y)` as a function of :math:`\phi_X(t_x)` given :math:`Y = X + a` where :math:`a` is a constant.

In terms of PDFs, if :math:`Y = X + a` then the joint PDF is given by 
math::
    f_{X,Y}(x,y) = f_X(x)\delta(y-x-a)
Taking the Fourier transform we have
math::
    \phi_{X,Y}(t_x,t_y) = \iint f_X(x)\delta(y-x-a)e^{-i(t_x x+t_y y)} \,dx\,dy
Solving the double integral, we obtain
math::
    \phi_{X,Y}(t_x,t_y) = e^{-i t_y a} \phi_X(t_x+t_y)

Addition of Two Random Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Our problem is to find the joint CHF :math:`\phi_{X,Y,Z}(t_x,t_y,t_z)` as a function of :math:`\phi_{X,Y}(t_x,t_y)` given :math:`Z = X + Y`.

In terms of PDFs, if :math:`Z = X + Y` then the joint PDF is given by 
math::
    f_{X,Y,Z}(x,y,z) = f_{X,Y}(x,y)\delta(z-x-y)
Taking the Fourier transform we have
math::
    \phi_{X,Y,Z}(t_x,t_y,t_z) = \iiint f_{X,Y}(x,y)\delta(z-x-y)e^{-i(t_x x+t_y y+t_z z)} \,dx\,dy\,dz
Solving the triple integral, we obtain
math::
    \phi_{X,Y,Z}(t_x,t_y,t_z) = \phi_{X,Y}(t_x+t_z,t_y+t_z)

This is a direct result of the central slice theorm.

Note that if we start with a joint chf with non-1st generation parents, then we work as if the parents were mutually independent with the sum, since we are already conditioning on the 1st generation parents.

For the subtraction of two random variables :math:`Z = X - Y`, we may similarly derive the expression 
math::
    \phi_{X,Y,Z}(t_x,t_y,t_z) = \phi_{X,Y}(t_x+t_z,t_y-t_z)

Multiplication of a Constant to a Random Variable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Our problem is to find the joint CHF :math:`\phi_{X,Y}(t_x,t_y)` as a function of :math:`\phi_X(t_x)` given :math:`Y = a X` where :math:`a` is a constant.

In terms of PDFs, if :math:`Y = aX` then the joint PDF is given by 
math::
    f_{X,Y}(x,y) = f_X(x)\delta(y-ax)
Taking the Fourier transform we have
math::
    \phi_{X,Y}(t_x,t_y) = \iint f_X(x)\delta(y-ax)e^{-i(t_x x+t_y y)} \,dx\,dy
Solving the double integral, we obtain
math::
    \phi_{X,Y}(t_x,t_y) = \phi_X(t_x + a t_y)


Computing and Plotting Results
------------------------------

Bayesian Inference
------------------
