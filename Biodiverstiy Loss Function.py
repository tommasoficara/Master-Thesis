# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 18:17:00 2021

@author: Tommaso
"""

import scipy.optimize
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


pt=[0,0.8, 2, 3, 4.3]
pe=[2.8*10**-4,0.028, 0.052, 0.085, 0.16]

k= 1.73 * 10**-2

fi=4.4*10**-3

teta= 2.8*10**-4



model_fit=scipy.optimize.curve_fit(lambda x, fi, k: (1+teta+(k*x)+(fi*x**2)),pt,pe, p0=(0.08, 0.25))


def K(x):
    result=[]
    for i in range(len(x)):
        result.append(teta+k*x[i]+fi*x[i]**2)
    return result



t=np.arange(0,6,0.1)
K(t)
plt.title('Kaushal Extinction from Climate Change')
plt.plot(t,K(t))

plt.legend()



sns.scatterplot(pt, pe)
model_poly=np.poly1d(np.polyfit(pt, pe,2))

def B(T):
    result=[]
    for i in range(len(T)):
        result.append(100*(model_poly[0]+model_poly[1]*T[i]+model_poly[2]*T[i]**2))
    return result


'''   
def Loss(T):
    result=[]
    for i in range(len(T)):
        result.append(1-(model_poly[0]+model_poly[1]*T[i]+model_poly[2]*T[i]**2))
    return result
'''

plt.figure()
plt.title('Extinction Predictions from Climate Change', fontsize=18)
#sns.scatterplot(pt, pe)
plt.plot(t, B(t), label='Urban (2015)')
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xlabel('Pre-industrial Temperature Rise (°C)', fontsize=14)
plt.ylabel('Percent Extinction',  fontsize=14)

#0.008432 x - 0.005579 x + 0.02761

#### THOMAS

pt=[0,1.25, 1.9, 2.5]
pe=[2.8*10**-4,0.18,0.245,0.3575]

#pt=[0,1.25, 1.9, 2.5]
#pe=[2.8*10**-4,0.09, 0.15, 0.21]

model_poly_t=np.poly1d(np.polyfit(pt, pe,2))

def B_t(T):
    result=[]
    for i in range(len(T)):
        result.append(teta+model_poly_t[1]*T[i]+model_poly_t[2]*T[i]**2)
    return result


plt.plot(t, B_t(t),label='Thomas et al. (2004)', color='red')
#sns.scatterplot(pt, pe)
plt.xlabel('Global mean temperature rise (°C)')
plt.ylabel('Biodiversity Loss (%)')
plt.fill_between(t, B_t(t),B(t), alpha=0.2, color='red')

plt.legend()

"""
pt=[0,1.25, 1.9, 3]
pe=[2.8*10**-4,0.09, 0.15, 0.21]


k= 0.07723237535565936 #1.73 * 10**-2 
fi=-0.0005016634384951093# 4.4*10**-3#
teta=2.8*10**-4



pt=[0,0.8, 2, 3, 4.3]
pe=[2.8*10**-4,0.028, 0.052, 0.085, 0.16]

k= 1.73 * 10**-2

fi=4.4*10**-3

teta= 2.8*10**-4

model_fit=scipy.optimize.curve_fit(lambda x, fi, k: (1+teta+(k*x)+(fi*x**2)),pt,pe, p0=(0.08, 0.25))

"""