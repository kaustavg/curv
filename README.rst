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

Computing and Plotting Results
------------------------------

Bayesian Inference
------------------
