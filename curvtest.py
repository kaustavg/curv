import curv as cv

# X = cv.Normal(1,2)
# Z = cv.Uniform(-5,-4) - cv.Uniform(2,3)
# cv.plotpd(Z,(-10,0))
# print(str(Z))


n = cv.Net()
A = n.normal(1,2)
B = n.uniform(3,5)
C = A + B
cv.plotPD(C)
cv.plotNet(n)
print(n.joint)


# To Do now: create joint from result of addition
