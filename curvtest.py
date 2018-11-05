import curv as cv

A = cv.normal(1,0.1)
B = cv.uniform(0,4)
C = -B
cv.plotNet()
# print(cv.E(C))
# D = C * A**2
# cv.plotCHF(C,(-10,10))
cv.plotPD(C,(-10,10))

# A = cv.Normal(1,1)
# cv.pdplot(D,-10,10)
