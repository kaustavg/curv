import curv as cv

A = cv.uniform(0,1)
B = A-A
C = 3 + (-A)
D = 3 - A

cv.plotNet()
print(cv.E(B))
print(cv.V(B))
# D = C * A**2
# cv.plotCHF(C,(-10,10))
cv.plotPD(B,(-10,10))
cv.plotPD(D,(-10,10))

# A = cv.Normal(1,1)
# cv.pdplot(D,-10,10)