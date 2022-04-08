import matplotlib.pyplot as plt # plot function is used
from scipy.interpolate import interp1d # plotting
import statsmodels.api as sm #statistical testing
x=[i/5.0 for i in range(30)] # x=0-30

y=[1,2,1,2,3,4,2,4,3,5,7,8,7,8,9,10,9,10,9,10,11,13,14,13,13,15,16,17,18,20]
lowess=sm.nonparametric.lowess(y,x)
lowess_x=list(zip(*lowess))[0]
lowess_y=list(zip(*lowess))[1]
f=interp1d(lowess_x,lowess_y,bounds_error=False)
xnew=[i/10.0 for i in range(100)]
ynew=f(xnew)
plt.plot(x,y,'o')
plt.plot(lowess_x,lowess_y,'*')
plt.plot(xnew,ynew,'-')
plt.show()