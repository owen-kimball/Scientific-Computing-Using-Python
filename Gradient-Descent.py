import numpy as np
#import matplotlib.pyplot as plt
from scipy.optimize import fsolve, fminbound, fmin
from scipy.special import gamma

# Problem 1

# x(t) = (11/6)(e^(-t/12)-e^-t)
# x'(t) = (11/6)((-1/12)e^(-t/12)+e^-t)

# Part a
x = lambda t: (11/6)*((np.e**(-t/12))-(np.e**-t))
xderiv = lambda t: (11/6)*(((-1/12)*np.e**(-t/12))+(np.e**-t))
negderiv = lambda t: -(11/6)*(((-1/12)*np.e**(-t/12))+(np.e**-t))
A0 = xderiv(1.5)

# Part b

t = np.linspace(0,6,1000)
# fig, ax = plt.subplots()
# ax.plot(t,x(t),color="red")
# ax.plot(t,xderiv(t),color="blue")
# fig.show()
A1 = fsolve(negderiv,3)[0]

# Part c

A2 = x(A1)

# Problem 2

# ax.plot(t,gamma(t),color="green")
A3 = fminbound(gamma,0.5,2)
A4 = gamma(A3)

# Problem 3

# Note: z[0] = x and z[1] = y
func = lambda x,y: (x**2+y-11)**2+(x+y**2-7)**2
funcp = lambda p: func(p[0],p[1])

# Part a

A5 = funcp([3,4])

# Part b

A6 = fmin(funcp,[-4,0])

# Part c

gradxy = lambda x,y: np.array([4*x**3-42*x+4*x*y+2*y**2-14,
                           4*y**3-26*y+4*x*y+2*x**2-22])
gradf = lambda p: gradxy(p[0],p[1])

# Part ci

A7 = gradf(A6)

# Part cii

A8 = np.linalg.norm(A7,2)

# Part d

def gradDesc(pInitial,gradFunc,f):
    n=0
    line = lambda t: pInitial-t*gradFunc(pInitial)
    line_heights = lambda t: f(line(t))
    minHeight = fminbound(line_heights,0,1)
    pInitial = gradFunc([minHeight,minHeight])
    n+=1
    tol = 10**-7
    while np.linalg.norm(gradFunc(pInitial))>tol:
        line = lambda t: pInitial-t*gradFunc(pInitial)
        line_heights = lambda t: f(line(t))
        minHeight = fminbound(line_heights,0,1)
        pInitial = line(minHeight)
        n+=1
    n=n-1
    return np.array([pInitial,[n,0]])

# Part di

p0 = np.array([-4,0])
A9 = gradDesc(p0,gradf,funcp)[0]
A10 = funcp(A9)

# Part dii
A11 = gradDesc(p0,gradf,funcp)[1][0]
