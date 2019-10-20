 # -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 23:58:47 2017
Financial Functions Lib
@author: Xiao Cheng, MS of Mathematics in Finance at NYU Cournat
This is a file containing useful functions as a Quant
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from math import e 
import scipy.stats as spst
import copy
#from scipy.stats import norm





# %%                 ==============
#                     Risk Session
#                    ==============
# %% Volatility measure

def downsideDeviation(se_return,MARR=0):
    '''
    Calculate Downside Deviation
    
    Parameters
    ----------
    se_return: list/array,price return array
    MARR: float(optional), minimum acceptable rate of return
    
    Returns
    -------
    downDevi: float, downside deviation
    '''
    se_return = np.asarray(se_return)
    temp = se_return[se_return < MARR]#Minimum Acceptable Rate of Return
    downDevi = (sum((MARR-temp)**2)/len(se_return))**0.5
    return downDevi

def meanAbsoluteDeviation(se_return):
    '''
    Calculate Mean Absolute Deviation, E[|X-E[X]|]
    
    Parameters
    ----------
    se_return: list/array,price return array
    
    Returns
    -------
    res: float, mean absolute deviation
    '''
    se_return = np.asarray(se_return)
    res = np.mean(np.abs(se_return - np.mean(se_return)))
    return res

def interPtileRange(se_return,p=0.1):
    '''
    Calculate inter-p-tile-range, F^(-1)(1-p)-F^(-1)(p)
    
    Parameters
    ----------
    se_return: list/array,price return array
    p: float, probability
    
    Returns
    -------
    res: float,inter-p-tile-range
    '''
    se_return = np.asarray(se_return)
    res = np.quantile(se_return,1-p) - np.quantile(se_return,p)
    return res

# %% Downside risk measure

def valueAtRisk(se_return,p=0.95,type='Sample'):
    '''
    Calculate VaR, inf⁡{x│Pr⁡(-X>x)≤1-p}
    
    Parameters
    ----------
    se_return: list/array,price return array
    p: float, probability
    type: string(Sample/Normal) type of calculation
    
    Returns
    -------
    res: float,VaR
    
    Note
    ----
    numpy percentile function does the wrong thing - assumes first observation 
    is 0 percentile and last is 100th. Cure that by adding extra observations 
    at each end. 
    
    '''
    se_return.sort()
    if type == 'Sample':
        se_return = copy.copy(se_return)
        se_return.append(min(se_return)-1)
        se_return.append(max(se_return)+1)
        se_return = np.asarray(se_return)
        res = -np.percentile(se_return,(1-p)*100)
    elif type == 'Normal':
        se_return = np.asarray(se_return)
        samp_mean=np.mean(se_return) 
        samp_std=np.std(se_return)
        res = -(samp_mean+samp_std*spst.norm.ppf(1-p)) 
    else:
        raise ValueError("Type not in ['Sample','Normal']")
    return res

def cVaR(se_return,p=0.95,type='Sample'):
    '''
    Calculate cVaR
    
    Parameters
    ----------
    se_return: list/array,price return array
    p: float, probability
    
    Returns
    -------
    res: float,cVaR
    type: string(Sample/Normal) type of calculation
    
    '''
    se_return.sort()
    if type == 'Sample':
        VaR = valueAtRisk(se_return,p,'Sample')
        se_return = np.asarray(se_return)
        nexceed = max(np.where(se_return<=-VaR)[0]) 
        #-VaRp is (1-p) of the way between y[nexceed] and y[nexceed+1] 
        res = -(np.sum([yy for yy in se_return if yy<=-VaR])-(1-p)*VaR)/(nexceed+2-p) 
    elif type == 'Normal':
        samp_mean = np.mean(se_return) 
        samp_std = np.std(se_return)    
        nVaR = valueAtRisk(se_return,p,'Normal')
        se_return = np.asarray(se_return)
        res = -samp_mean+samp_std*np.exp(-.5*(nVaR/samp_std)**2)/((1-p)*np.sqrt(2*np.pi)) 
    else:
        raise ValueError("Type not in ['Sample','Normal']")
    return res    


def expectedShortfall(se_return,p=0.05):
    '''
    Expected shortfall
    
    Parameters
    ----------
    se_return: list/array,price return array
    p: float, probability
    
    Returns
    -------
    res: float,Expected shortfall
    
    Note
    ----
    May not be precise, use cVaR instead 
    
    '''
    se_return = np.asarray(se_return)
    res = -se_return[(se_return<=np.quantile(se_return,p))].mean()
    return res



# %%                 ================
#                     Pricing Session
#                    ================
# %% 
    













# %% Build Interest Rate Curves

class BuildFCurve(object):
    '''
    Build F curve and Z curve from market LIBOR & Swap rate quotation(Derivative Securities)
    with optimization of the loss function
    Note: All rate F S LIBOR denote as %
    
    Parameters
    ----------
    data_dict: dict<string,list>,
                market LIBOR & Swap rate quotation, 'time' refers to time intervals
                with unit month, 'rate' refers to the market rate with %
    granularity: int
                granularity of interpolation and optimization
    penaltyPara: tuple<float>
                penalty parameters of a, b in loss function for Brute Force optimization                
                
    Methods
    -------
    interpolate_linear: list
                make linear interpolation of the market rate
    interpolate_NelsonSiegel: list
                make interpolation based on Nelson-Siegel smoothed method (plot)
    optimize: list
                Use Brute Force to optimize and get F curve
    HaganWest_iteration: list,list,list
                Use Hagan-West iteration bootstraping method to get F and 
                yield curve (plot)
    getDataFrame: pandas.DataFrame,pandas.DataFrame
                get information of the Brute Force optimization result and plot
    
    Examples
    --------
        # interest rate(LIBOR & SWAPS)
        >>> rateList = [0.19,0.33,0.46,0.64,0.97,1.26,1.48,1.66,1.94,2.21,2.67] 
        >>> timeList =  [1,3,6,12,24,36,48,60,84,120,360] # months
        >>> data_dict = data_dict={'time':timeList,'rate':rateList}
        # build class
        >>> x = BuildFCurve(data_dict,granularity=3,penaltyPara=(1,1))
        # optimize
        >>> result = x.optimize()
        # get result
        >>> df,df1 = x.getDataFrame()
        # HaganWest_iteration
        >>> x.HaganWest_iteration()
    '''
    def __init__(self,data_dict={'time':[1,3,6,12,24,36,48,60,84,120,360],'rate':[]},granularity=1,penaltyPara=(1,1)):
        '''constructor'''
        self.f_result = None                                                 # optimize result
        self.referTime, self.referRate = data_dict['time'],data_dict['rate'] # list of time and corresponding list of market rate 
        self.granularity = granularity                                       # granularity, # of months
        self.penalty_a,self.penalty_b = penaltyPara                          # penalty function parameters
        # generate the whole time span according to granularity
        self.timeList = list(range(0,361,granularity))                       # total time list
        if self.timeList[1] != 1:
            self.timeList[0] = 1
        else:
            self.timeList.remove(0)
    
    def interpolate_linear(self):
        '''use linear interpolation to get the start value'''
        f_interpolate = []
        pointer = 0
        for i in self.timeList:
            if i in self.referTime:
                f_interpolate.append(self.referRate[self.referTime.index(i)])
                pointer = self.referTime.index(i)
            else:
                slop = (self.referRate[pointer+1] - self.referRate[pointer])/(self.referTime[pointer+1]-self.referTime[pointer])
                interest = self.referRate[pointer] + (i - self.referTime[pointer]) * slop
                f_interpolate.append(interest)
        return f_interpolate
    
    def interpolate_NelsonSiegel(self):
        '''use Nelson-Siegel smoothed version to interpolate (Risk&Port)'''
        self.linear_f = self.interpolate_linear() # share data
        # set start
        beta0_start = self.referRate[-1] # set price[-1]
        beta1_start = self.referRate[0]-self.referRate[-1] # set price[0] - price[-1]
        beta2_start = 0
        tau_start = 1 
        # begin optimize
        x0 = np.array([beta0_start,beta1_start,beta2_start,tau_start])
        res = minimize(self.__mad, x0, method='nelder-mead',options={'xtol': 1e-8, 'disp': True, 'maxiter': 1000})
        beta0,beta1,beta2,tau = res.x        
        NS_curve = self.__getNSCurve(beta0,beta1,beta2,tau)
        # plot
        plt.plot(self.timeList,NS_curve,label='Nelson-Siegel method')
        plt.plot(self.timeList,self.linear_f,label='Linear method')
        plt.xlabel('Tenor')
        plt.ylabel('Yield Curve')
        plt.legend()        
        return NS_curve
        
    def optimize(self):
        '''optimize function(Brute force)'''
        # set start
        #fList_start = np.array([0] * len(self.timeList))
        fList_start = self.interpolate_linear() # use linear interpolation as initial guess
        # begin optimize
        res = minimize(self.__penaltyFunc, fList_start, method='nelder-mead',options={'xtol': 0.01, 'disp': True, 'maxiter':20000})
        optimize_result = res.x
        self.f_result = optimize_result
        return optimize_result       
    
    def  HaganWest_iteration(self):
        ''' Hagan-West iteration method,bootstraping
        /Interpolation Methods for Curve Construction/,
          2006,PATRICK S.HAGAN & GRAEME WEST, P92
        '''
        yieldCurve0 = np.asarray(self.interpolate_linear()) # initial guess of the yield curve
        zCurve = np.exp(- yieldCurve0/100 * self.timeList/12) # initial guess of the discount curve
        swapCurve = np.asarray(self.interpolate_linear()) # swap curve interpolation
        yieldCurve = [yieldCurve0[0]] # iteration result
        for i,swap in enumerate(swapCurve):
            if i == 0:
                continue
            else:
                # rn is not scaled
                rn = - np.log((1 - np.sum(zCurve[:i])*self.granularity/12 * swap/100)/(1+ swap/100 * self.granularity/12))/(self.timeList[i]/12)
                yieldCurve.append(rn*100)
                zCurve[i] = np.exp(-rn * self.timeList[i]/12) # update Zi
        # forward rate, not smooth
        fCurve = (zCurve[:-1]/zCurve[1:] -1)/(self.granularity/12) * 100
        # plot
        plt.plot(self.timeList,yieldCurve,label='Yield Curve')
        plt.plot(self.timeList,zCurve,label='Z')
        #plt.plot(self.timeList[:-1],fCurve,label='F')
        plt.xlabel('number of months')
        plt.ylabel('rate %')        
        plt.legend()
        plt.show()
        return zCurve,yieldCurve,fCurve

    def getDataFrame(self):
        zList,zList_refer = self.__constructZ(self.f_result)
        zList_real = self.__constructRealZ(zList)
        dic = {'time':self.timeList,'F':self.f_result,'Z':zList}
        dic1 = {'time':self.referTime,'Z':zList_refer,'Z_swap':zList_real}
        df = pd.DataFrame(dic)
        df1 = pd.DataFrame(dic1)
        # plot
        plt.plot(self.timeList,self.f_result,label='F')
        plt.plot(self.timeList,zList,label='Z')
        plt.xlabel('number of months')
        plt.ylabel('rate %')        
        plt.legend()
        plt.show()
        return df,df1
    
    # Auxillary functions
    def __penaltyFunc(self,fList):
        '''
        penalty function
        min: 
            sum[(Zi-Zi_real)^2] 
            + a * sum[(F_(i+1) + F_(i-1) -2Fi)^2] 
            + b * sum[(F_(i+1)-F_(i-1))^2]
        '''
        result = 0
        zList,zList_refer = self.__constructZ(fList)
        zRealList_refer = self.__constructRealZ(zList)
        # item for error
        result += np.sum((zList_refer-zRealList_refer)*(zList_refer-zRealList_refer))
        # item for penalty
        fList = np.asarray(fList)
        mid = fList[1:][:-1] # Fi
        plus = fList[2:]     # F_(i+1)
        lag = fList[:-2]     # F_(i-1)                
        result += self.penalty_b * np.sum((plus-lag) * (plus-lag)) # item b
        array = plus + lag - 2* mid
        result += self.penalty_a *np.sum(array * array) # item a
        print("Penalty Value: %.5f"%result)
        return result    
    
    def __constructZ(self,fList):
        '''
        construct Z according to F
        Z(T2)=Z(T1)/(1+F*dT)
        '''
        zList = [] # all points
        # first two points
        z0 = 1/(1+fList[0]*0.01/12)
        z1 = z0/(1+fList[1]*0.01*(self.timeList[1]-self.timeList[0])/12)
        zList.append(z0)
        zList.append(z1) # since the interval may not be the same
        # loop for the rest
        for i in range(2,len(fList)):
            ztmp = zList[-1]/(1+fList[i] * 0.01 * self.granularity/12)
            zList.append(ztmp)
        zList_refer = [zList[self.timeList.index(j)] for j in self.referTime] # selected points
        return zList,np.asarray(zList_refer)

    def __constructRealZ(self,zList):
        '''
        construct Z according to S
        1-Zi=St*Ai, semi-annuity
        '''
        saList = np.where(np.asarray(self.timeList)%6 != 0,0,1) # semi-annuity
        anList = (np.asarray(zList) * saList).cumsum()/2 # Ai=sum(Zi*dT)
        # modify annuity for 1-6 months
        for i,item in enumerate(self.timeList):
            if item >= 6:
                break
            else:            
                anList[i] = zList[i]*item/12 # Ai=Zi*Ti/12
        anList_refer = [anList[self.timeList.index(j)] for j in self.referTime] # selected points
        # get real Zi
        zRealList_refer = []
        for i,rate in enumerate(self.referRate):
            z = 1-anList_refer[i] * rate/100 # Zi=1-St*Ai
            zRealList_refer.append(z)
        return np.asarray(zRealList_refer)    

    def __NS_r(self,f,beta0,beta1,beta2,tau):
        '''Calculate r(0,f) for Nelson-Siegel'''
        sumV = 0
        sumV += beta0
        sumV += beta1 * tau/f * (1-e**(-f/tau))
        sumV += beta2 * tau/f * (1-e**(-f/tau)*(1+f/tau))
        return sumV  
     
    def __getNSCurve(self,beta0,beta1,beta2,tau):
        '''get Nelson-Siegel Curve'''
        resultList = []
        for item in self.timeList:
            resultList.append(self.__NS_r(item,beta0,beta1,beta2,tau))
        return resultList

    def __mad(self,x):
        '''Calculate MAD for loss function of Nelson-Siegel'''
        mad = 0
        beta0,beta1,beta2,tau = x
        for i,f in enumerate(self.timeList):
            linear_interest = self.linear_f[i]
            ns_interest = self.__NS_r(self.timeList[i],beta0,beta1,beta2,tau)
            mad += abs(linear_interest - ns_interest)
        return mad        
        
# %%
        
        
        
        





