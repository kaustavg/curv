import curv as cv

A = cv.uniform(0,1)
B = cv.uniform(0,4)
C = A + B
D = C + A
cv.plotNet()
# print(cv.E(C))
# D = C * A**2
# cv.plotCHF(C,(-10,10))
cv.plotPD(C,(-10,10))
cv.plotPD(D,(-10,10))

# A = cv.Normal(1,1)
# cv.pdplot(D,-10,10)