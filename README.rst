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

    A = cv.Normal(1,2)
    B = cv.Uniform(-5,4)
    C = 4 + A - 2*B
    D = C * A**2
    cv.pdplot(D,-10,10)
    cv.netplot()
    A = cv.Normal(1,1)
    cv.pdplot(D,-10,10)

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

Addition of a Constant to a Continuous Random Variable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Our problem is to find the joint CHF :math:`\phi_{X,Y}(u,v)` as a function of :math:`\phi_X(u)` given :math:`Y = X + a` where :math:`a` is a constant.

The addition of two chfs in a joint distribution can be accomplished without integration using central slice theorm. The result of a sum of two RVs in a joint chf will be a slice of the joint chf of the parents. 

When forming the new joint chf, the slice will be placed along the new dimension in the negative direction. For example, if :math:`\frac{ \sum_{t=0}^{N}f(t,k) }{N}` then we first obtain the value of CHF(x,y) along the line y=x, then place those values in the new joint CHF(x,y,z) along the line x=y=-z. 

Note that if we start with a joint chf with non-1st generation parents, then we work as if the parents were mutually independent with the sum, since we are already conditioning on the 1st generation parents.

Computing and Plotting Results
------------------------------

Bayesian Inference
------------------
