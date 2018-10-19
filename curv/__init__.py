#!/usr/bin/env python

from .cvrv import *
from .cvnet import *
from .cvdistributions import *
from .cvoperations import *
from .cvplot import *
## BAYESIAN NETWORK
# Consider the following: A and B are both primitive RVs. Then define:
# C = A + B
# Since A and B are primitives and therefore independent, C's pdf is the
# convolution of A and B's pdfs. Now consider the following:
# D = A + C
# This is not a straightforward convolution of the pdfs, since C is not 
# independent with A. Instead, this is a projection of the joint pdf 
# perpendicular to the y=x line. Since we are internally storing the 
# characteristic functions, this projection is equivalent to slicing the
# joint cfun along the line y=x and scaling the result (by central-slice
# theorm).
