import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.optimize

# Problem 1

# Problem 1a

salmon_data = pd.read_csv('salmon_data.csv')
dfSalmon = pd.DataFrame(data=salmon_data)
newDfSalmon1 = dfSalmon[dfSalmon['Year']<=2024]
newDfSalmon2 = newDfSalmon1.dropna()
A0 = pd.to_numeric(newDfSalmon2["Chinook"])

# Problem 1b

chinook = A0 # A better reference
years = pd.to_numeric(newDfSalmon2["Year"])
polyfitLin = np.polyfit(years,chinook,1)
slopeLin = polyfitLin[0]
yintLin = polyfitLin[1]
A1 = np.array([slopeLin,yintLin]) # If wrong, it doesn't want the yint
# To also be a coeff - adapt this for the others as well!

# Problem 1c

polyfitCub = np.polyfit(years,chinook,3)
zeroArrCub = np.zeros(4)
for i in range(4):
    zeroArrCub[i] = polyfitCub[i]
A2 = zeroArrCub

# Problem 1d

polyfit5 = np.polyfit(years,chinook,5)
zeroArr5 = np.zeros(6)
for i in range(6):
    zeroArr5[i] = polyfit5[i]
A3 = zeroArr5

# Problem 1e

linCalc1 = np.polyval(polyfitLin,2025)
cubCalc1 = np.polyval(polyfitCub,2025)
calc5calc1 = np.polyval(polyfit5,2025)
f = lambda x: np.abs(x-632461)/632461
A4 = np.array([f(linCalc1),f(cubCalc1),f(calc5calc1)])

# Problem 1f

linCalc2 = np.polyval(polyfitLin,2030)
cubCalc2 = np.polyval(polyfitCub,2030)
calc5calc2 = np.polyval(polyfit5,2030)
A5 = np.array([(linCalc2),(cubCalc2),(calc5calc2)])

# Problem 1g

spline = scipy.interpolate.CubicSpline(years,chinook)
A6 = spline(2020.75)

# Problem 2

# Problem 2a

co2_data = pd.read_csv('CO2_Data.csv')
dfCO2 = pd.DataFrame(data=co2_data)
A7 = dfCO2['years after 1958'].to_numpy()
A8 = dfCO2['CO2 average'].to_numpy()

# Problem 2b

years = A7
co2_avg = A8
co2_avgGuess = np.zeros(A8.size)
a = 280
b = 40
r = 0.15
def func(arr):
    squaredErrors = 0
    y=lambda t: arr[0]+arr[1]*np.exp(arr[2]*t)
    for i in range(years.size):
        squaredErrors += (y(years[i])-co2_avg[i])**2
    return squaredErrors
param_Arr = np.array([a,b,r])
A9 = func(param_Arr)

# Problem 2c

A10 = scipy.optimize.fmin(func, param_Arr)

# Problem 2d

co2_avgGuess2 = np.zeros(A8.size)
a = 280
b = 40
r = 0.15
c = 5
d = 6
e = 0
def func2(arr):
    squaredErrors = 0
    y=lambda t: arr[0]+arr[1]*np.exp(arr[2]*t)+arr[3]*np.sin(arr[4]*(t-arr[5]))
    for i in range(years.size):
        squaredErrors += (y(years[i])-co2_avg[i])**2
    return squaredErrors
param_Arr = np.array([a,b,r,c,d,e])
A11 = func2(param_Arr)

# Problem 2e

param_Arr=np.array([A10[0],A10[1],A10[2],c,d,e])
A12 = scipy.optimize.fmin(func2,param_Arr,maxiter=2000)
