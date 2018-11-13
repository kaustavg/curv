import curv as cv

A = cv.normal(1,2)
B = cv.uniform(-5,4)
C = 4 + A - 2*B

cv.plotNet()
print(cv.E(C))
print(cv.V(C))
# D = C * A**2
# cv.plotCHF(C,(-10,10))
cv.plotPDF(C,(-10,10))
