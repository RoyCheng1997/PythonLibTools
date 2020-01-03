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
from math import e 
import scipy.stats as spst
from scipy.stats import t
import scipy.optimize as scpo
from scipy.optimize import minimize_scalar,minimize
from scipy import stats
import copy
import random
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



# %%                 ======================
#                     Fixed Income Session
#                    ======================
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
    interpolate_shortRate: list,list,list
                make linear interpolation of the market spot rate and get 
                implied short rate curve (plot)
    optimize: list
                Use Brute Force to optimize and get F curve
    HaganWest_iteration: list,list,list
                Use Hagan-West iteration bootstraping method to get F and 
                yield curve (plot)
    HullWhite_simulation: list,list
                Use Hull-White stochastic simulation to get sample spot curve 
                and sample short rate curve
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
    
    def interpolate_shortRate(self):
        '''
        use monthly linear interpolation for spot(market) rate and
        bootstrap an implied short rate curve (Risk&Port)
        short rate: R=r(t,t)=lim r(t,t+dt) instantaneous forward rates 
        i.e. the 5-year point on the short rate curve is the annualized rate 
        from 5 years to 5 years plus one microsecond from now.
        '''   
        tenors_in = list(np.asarray(self.referTime)/12)
        curve_in = self.referRate
        curve_out=[]
        tenors_out=[]
        shortrates=[]
        idxin=0
        mnthin=round(tenors_in[idxin]*12)
        months=round(tenors_in[len(tenors_in)-1]*12)
        #Fill in curve_out every month between the knot
        #points given in curve
        #As curve is filled in, bootstrap a short rate curve
        for month in range(int(months)):
            tenors_out.append(float(month+1)/12)
            if (month+1==mnthin):   #Are we at a knot point?
                #Copy over original curve at this point
                curve_out.append(curve_in[idxin])
                #Move indicator to next knot pratematrix[0]oint
                idxin+=1
                if (idxin!=len(tenors_in)):
                    #Set month number of next knot point
                    mnthin=round(tenors_in[idxin]*12)
            else:   #Not at a knot point - interpolate
                timespread=tenors_in[idxin]-tenors_in[idxin-1]
                ratespread=curve_in[idxin]-curve_in[idxin-1]
                if (timespread<=0):
                    curve_out.append(curve_in[idxin-1])
                else:
                    #compute years between previous knot point and now
                    time_to_previous_knot=(month+1)/12-tenors_in[idxin-1]
                    proportion=(ratespread/timespread)*time_to_previous_knot
                    curve_out.append(curve_in[idxin-1]+proportion)
            #Bootstrap a short rate curve
            short=curve_out[month]    
            if (month!=0):
                denom=tenors_out[month]-tenors_out[month-1]
                numer=curve_out[month]-curve_out[month-1]
                if (denom!=0):
                    short+=tenors_out[month]*numer/denom
            shortrates.append(short)
        # plot
        plt.plot(tenors_out, curve_out, label='spot rate') 
        plt.plot(tenors_out, shortrates, label='short rate') 
        plt.xlabel('Tenor (years)') 
        plt.ylabel('Rate (%/year)') 
        plt.ylim(0,max(curve_out)+.5) 
        plt.legend() 
        plt.grid(True) 
        plt.show()
        return(tenors_out,curve_out,shortrates)    
            
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
    
    def HullWhite_simulation(self,xlambda=1,sigma=.05):
        '''
        a Hull-White randomly generated short-rate curve
        and a yield curve integrating the Hull-White short curve
        
        Ornstein-Uhlenbeck process: dR = \lambda*(R_infi-R)*dt+\sigma*d(\beta)
        xlambda is spring stiffness; sigma is volatility
        '''
        tenors,curvemonthly,shortrates = self.interpolate_shortRate()
        #random.seed(3.14159265)
        randomwalk=[]
        curvesample=[]
        for i,rate in enumerate(shortrates):
            if i==0: # initialize
                randomwalk.append(shortrates[i])
                curvesample.append(randomwalk[i])
            else:
                deterministic=xlambda*(shortrates[i]-randomwalk[i-1])
                #multiply by delta-t
                deterministic*=(tenors[i]-tenors[i-1])
                stochastic=sigma*random.gauss(0,1)
                randomwalk.append(randomwalk[i-1]+deterministic+stochastic)
                #sample curve is average of short rate
                #random walk to this point
                cs=curvesample[i-1]*i
                cs+=randomwalk[i]
                cs/=(i+1)
                curvesample.append(cs)
        #Plot the four curves      
        plt.plot(tenors, curvemonthly, label='spot rate curve')
        plt.plot(tenors, shortrates, label='Implied short rate curve')
        plt.plot(tenors, randomwalk, label='Sample short rate curve $R(\omega)$')
        plt.plot(tenors, curvesample, label='Sample spot rate curve $\overline{R}(\omega)$')
        ## Configure the graph
        plt.title('Hull-White Curve Generation')
        plt.xlabel('Tenor (years)')
        plt.ylabel('Rate (%/year)')
        plt.legend()
        plt.grid(True)
        plt.show()      
        return (curvesample,randomwalk) # spot/short

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
        
# %%                 ====================
#                         Test Session
#                    ====================
def hotelling(T1,T2,p,m1,m2,s1,s2):
    # Equality of mean vectors: Hotelling's Test
    #Compute Hotelling's statistic and p value
    #Combined covariance matrix
    scomb=((T1-1)*s1+(T2-1)*s2)/(T1+T2-2)
    #Multiplier for statistic
    hmult=(T1+T2-p-1)*T1*T2/((T1+T2-2)*p*(T1+T2))
    #Matrix algebra for statistic
    if p==1:
        h12=hmult*(m1-m2)**2/scomb
    else:
        h12=hmult*np.matmul(np.matmul(m1-m2,np.linalg.inv(scomb)),m1-m2)
    p_value = 1 - spst.f.cdf(h12, p, T1+T2-1-p)
    #Note when the dimension p=1, p_value will equal spst.ttest_ind(x1,x2)
    return(h12,p_value)    

def levene(T1,T2,x1,x2):
    # Equality of variances: Levene's Test
    # Apply Levene's Test to three-currency example with previous years compared to latest year
    #Note the results shown are the same as
    #scipy.stats.levene(d[:prev_year_n,k],d[prev_year_n:,k],center='mean')    
    m1=np.average(x1)
    m2=np.average(x2)
    z1j=[np.abs(x1[j]-m1) for j in range(T1)]
    z2j=[np.abs(x2[j]-m2) for j in range(T2)]
    z1=np.average(z1j)
    z2=np.average(z2j)
    levene_mult=(T1+T2-2)*T1*T2/(T1+T2)
    levene_denom=np.sum((z1j-z1)**2)+np.sum((z2j-z2)**2)
    levene_stat=levene_mult*(z1-z2)**2/levene_denom
    p_value = 1 - spst.f.cdf(levene_stat, 1, T1+T2-2)
    return(levene_stat,p_value)    

def BoxM(T1,T2,s1,s2):
    # Equality of covariance matrices: Box's M Test
    # Box M Test for covariance matrices
    # From G.E.P. Box, "A General Distribution Theory for a Class of Likelihood Criteria",
    # Biometrika 36, December 1949, pp. 317-346.    
    #Tests for equality of two covariance matrices, s1 and s2
    #T1 and T2 are numbers of observations for s1 and s2
    #Returns M statistic and p-value
    #Make sure dimension is common
    if len(s1)!=len(s2):
        print("Error: different dimensions in Box M Test:",len(s1),len(s2))
        return(0,0)
    #Matrices are pxp
    p=len(s1)
    #Form the combined matrix
    scomb=(T1*s1+T2*s2)/(T1+T2)
    #Box M statistic
    Mstat=(T1+T2-2)*np.log(np.linalg.det(scomb))-(T1-1)*np.log(np.linalg.det(s1))-(T2-1)*np.log(np.linalg.det(s2))
    #Multipliers from equation (49) in Box 1949.
    A1=(2*p**2+3*p-1)/(6*(p+1))
    A1*=(1/(T1-1)+1/(T2-1)-1/(T1+T2-2))
    A2=(p-1)*(p+2)/6
    A2*=(1/(T1-1)**2+1/(T2-1)**2-1/(T1+T2-2)**2)
    discrim=A2-A1**2
    #Degrees of freedom
    df1=p*(p+1)/2
    if discrim <= 0:
        #Use chi-square (Box 1949 top p. 329)
        test_value=Mstat*(1-A1)
        p_value=1-spst.chi2.cdf(test_value,df1)
    else:
        #Use F Test (Box 1949 equation (68))
        df2=(df1+2)/discrim
        b=df1/(1-A1-(df1/df2))
        test_value=Mstat/b
        p_value=1-spst.f.cdf(test_value,df1,df2)
    return(test_value,p_value)    

def testNormal(array):
    # QQ plot
    mean = np.mean(array)
    stdev = np.std(array)
    nobs = len(array)
    x = stats.norm.ppf([i/(nobs+1) for i in range(1,nobs+1)])
    #Plot the diagonal
    line=plt.plot(x, x)
    plt.setp(line, linewidth=2, color='r')
    #Plot the actuals
    y = np.sort(np.array((array-mean)/stdev))
    plt.scatter(x, y, s=40, c='g')    
    ## Configure the graph
    plt.title('Q-Q plot')
    plt.xlabel('Standardized Inverse Normal')
    plt.ylabel('Standardized Observed Log-return')
    plt.grid(True)
    plt.show
    # PP plot
    x=[i/(nobs+1) for i in range(1,nobs+1)]
    #Plot the diagonal
    line = plt.plot(x, x)
    plt.setp(line, linewidth=2, color='r')
    #Plot the actuals
    y=np.sort(np.array(stats.norm.cdf((array-mean)/stdev)))
    plt.scatter(x, y, s=40, c='g')
    ## Configure the graph
    plt.title('P-P plot')
    plt.xlabel('Fraction between 0 and 1')
    plt.ylabel('Normal CDF of Standardized Observed Log-return')
    plt.grid(True)
    plt.show
    #Jarque-Bera
    sk = stats.skew(array)
    ku = stats.kurtosis(array)    #This gives excess kurtosis
    jb = (nobs/6)*(sk**2+(ku**2)/4)
    chi2 = stats.chi2.cdf(jb,2)
    print('Skewness %f' % sk)
    print('Excess Kurtosis %f' % ku)
    print('Jarque-Bera Statistic %f' % jb)
    print('Chi-squared probability non-normal %f' % chi2)


# %%                 ====================
#                    Distribution Session
#                    ====================
# %% 

def distribution_studentT(dList=[5,6,7,1000]):
    '''parameter d in t distribution'''
    highend=[6.,-3.]
    title_strings=["Overall view of Student\'s T densities",
                   "Left cumulative tails of Student\'s T distributions"]
    plt.figure(figsize=(12,4))
    for sp, top in enumerate(highend):
        plt.subplot(1, 2, sp+1)
        x = np.linspace(-6., top, 100)
        for d in dList:
            if sp==0:
                plt.plot(x, t.pdf(x, d), lw=3, alpha=0.6, label=str(d))
            else:
                plt.plot(x, t.cdf(x, d), lw=3, alpha=0.6, label=str(d))
        plt.grid()
        plt.legend()
        plt.title(title_strings[sp])
    plt.tight_layout()
    plt.show()
    
def distribution_mixNormal(yList=[5,10,15,20]):
    #Generate mixed normal kurtosis graph
    #x contains fractions of the riskier distribution going into the mix
    x = np.arange(.05,.5,.05)
    #y contains the multiple (how much riskier the riskier distribution is than the less risky)  
    y = yList
    z=np.zeros((len(y),len(x)))
    for i,multiple in enumerate(y): # multiple=r
        for j,mixamount in enumerate(x): # mixamount=w1
            #numerator
            z[i,j] = mixamount*multiple**2
            z[i,j] = mixamount*multiple**2+1-mixamount
            #denominator
            z[i,j] /= (mixamount*multiple+1-mixamount)**2
            #multiply by 3 and subtract 3
            z[i,j] -= 1
            z[i,j] *= 3
        plt.plot(x,z[i,:],label=str(multiple))
    plt.grid()
    plt.legend()
    plt.xlabel('Fraction of riskier distribution, '+r'$w_1$')
    plt.ylabel('Kurtosis of mix')
    plt.title('Kurtosis of mixtures of normals')
    plt.show()
            
def distribution_stable(alphaList=[1.1,1.5,1.7,2]):
    highend=[6.,-3.]
    beta = 0     #symmetric
    title_strings=["Overall view of stable distributions",
                   "Left cumulative tails of stable distributions"]
    plt.figure(figsize=(12,4))
    for sp, top in enumerate(highend):
        plt.subplot(1, 2, sp+1)
        x = np.linspace(-6., top, 100)
        for alpha in alphaList:
            if sp==0:
                plt.plot(x, spst.levy_stable.pdf(x, alpha, beta),
                    label=str(alpha))
            else:
                plt.plot(x, spst.levy_stable.cdf(x, alpha, beta),
                    label=str(alpha))
        plt.grid()
        plt.legend()
        plt.title(title_strings[sp])
    plt.tight_layout()
    plt.show()
        
def distribution_extremeValue(gamma_list=[-0.5,0.0,.5],x_list=np.arange(-2.5,3.55,.05)):
    name_list=['Weibull','Gumbel','Frechet'] #Generate Weibull, Gumbel and Frechet densities
    for i,gamma in enumerate(gamma_list):
        y=[gev(gamma,x) for x in x_list]
        plt.plot(x_list,y,label=name_list[i])
    plt.grid()
    plt.legend()
    plt.title('Generalized Extreme Value Densities')
    plt.show()     
    
def gev(gamma,x):
    #Return pdf of a gev with parameter gamma at point x
    if gamma != 0:
        op_gamma_x = 1 + gamma*x
        if op_gamma_x > 0:    #make sure in the support
            oo_gamma_power=op_gamma_x**(-1/gamma)
            gev_cdf=np.exp(-oo_gamma_power)
            gev_pdf=oo_gamma_power*gev_cdf/op_gamma_x
        else:
            gev_pdf = np.nan
    else:   #gumbel
        gev_cdf = np.exp(-np.exp(-x))
        gev_pdf = np.exp(-x)*gev_cdf
    return(gev_pdf)

def distribution_tail():
    #Generate Generalized Pareto densities
    xlist=np.arange(0,3.05,.05)
    #gamma=0; density is exp(-x)
    plt.plot(xlist,[np.exp(-x) for x in xlist],label='0')
    #gamma=-.5; upper limit is 2
    shortlist=np.arange(0,2.05,.05)
    y=[1-x/2 for x in shortlist]
    y+=[np.nan]*(len(xlist)-len(shortlist))
    plt.plot(xlist,y,label='-.5')
    #gamma=-1; constant value of 1 up to 1
    shortlist=np.arange(0,1.05,.05)
    y=[1]*len(shortlist)
    y+=[np.nan]*(len(xlist)-len(shortlist))
    plt.plot(xlist,y,label='-1')
    #gamma=-2; blows up at x=.5
    #1/sqrt(1-2x)
    shortlist=np.arange(0,.5,.05)
    y=[1/(1-2*x)**.5 for x in shortlist]
    y+=[np.nan]*(len(xlist)-len(shortlist))
    plt.plot(xlist,y,label='-2',color='y')
    plt.grid()
    plt.legend()
    plt.title('Generalized Pareto densities')
    plt.show()

def estimate_tailParameters_example():
    threshold=2#Take exceedances over threshold; show number and average exceedance
    n_sample=10000#Generate 10,000 standard normal draws
    random.seed(314159)
    #Using this random number generator because we can control the seed
    sample=[]
    for i in range(n_sample):
        sample.append(random.gauss(0,1))
    exceeds=[s-threshold for s in sample if s>threshold]
    numex=len(exceeds)    #exceedance count
    avex=np.mean(exceeds) #y-bar
    #Maximum likelihood function 6.87 with these values
    maxlike=-numex*(np.log(avex)+1)
    print('Number of exceedances over {0}: {1}'.format(threshold,numex))
    print('Expected exceedances: {0}'.format(int(n_sample*spst.norm.cdf(-2)+.5)))
    print('Average exceedance:',avex)
    print('Maximum likelihood function at beta:',maxlike)
    #initial guess for parameters is gumbel
    init_params=[0.,np.log(avex)]
    #Log-max-likelihood for GPD. Sign is reversed since we are using minimize.
    def gpd_lml(params):
        gma,beta_log=params
        #enforce positive beta
        beta=np.exp(beta_log)
        #check if gamma=0
        tolerance=10**(-9)
        if abs(gma)<tolerance:
            return(numex*(beta_log+avex/beta))
        #uses "exceeds" vector computed above
        log_sum=0
        #sum ln(1+gamma*yi/beta) when positive
        for i in range(numex):
            arg_log=1+gma*exceeds[i]/beta
            if arg_log<=0:
                log_sum+=1000*np.sign(1+1/gma)  #put a very discouraging amount in the sum
            else:
                log_sum+=np.log(arg_log)
        #scale
        log_sum*=(1+1/gma)
        if beta<=0:
            log_sum+=1000
        else:
            log_sum+=numex*np.log(beta)
        return(log_sum)   
    #Run the minimization.
    results = scpo.minimize(gpd_lml,
                                init_params,
                                method='CG')
    gma_res,beta_res=results.x
    beta_res=np.exp(beta_res)   #move back from log space
    print("Optimal gamma:",gma_res)
    print("Optimal beta:",beta_res)
    print("Support cap mF:",-beta_res/gma_res)
    print("Optimal LML:",-results.fun)
    #Show the CDF plot
    #x's are sorted values of the exceedances
    xsample=np.sort(exceeds)
    ysample=[(i+1)/(numex+1) for i in range(numex)]
    ygumbel=[1-np.exp(-x/avex) for x in xsample]
    yfitted=[1-(1+gma_res*x/beta_res)**(-1/gma_res) for x in xsample]
    plt.plot(xsample,ysample,label='Sample')
    plt.plot(xsample,ygumbel,label='Gumbel')
    plt.plot(xsample,yfitted,label='Fitted')
    plt.grid()
    plt.legend()
    plt.title('Figure 6.15: cdfs of empirical, Gumbel, and fitted distributions')
    plt.show
    
# %%                 ==================
#                    Volatility Session
#                    ==================
# %% 
# See 'stats_timeSeries_lib' for more time series functions
def Garch11Fit(initparams,InputData):
    import scipy.optimize as scpo
    import numpy as np
    #Fit a GARCH(1,1) model to InputData using (8.42)
    #Returns the triplet a,b,c (actually a1, b1, c) from (8.41)
    #Initial guess is the triple in initparams

    array_data=np.array(InputData)

    def GarchMaxLike(params):
        import numpy as np        
        #Implement formula 6.42
        xa,xb,xc=params
        if xa>10: xa=10
        if xb>10: xb=10
        if xc>10: xc=10
        #Use trick to force a and b between 0 and .999;
        #(a+b) less than .999; and c>0
        a=.999*np.exp(xa)/(1+np.exp(xa))
        b=(.999-a)*np.exp(xb)/(1+np.exp(xb))
        c=np.exp(xc)
        t=len(array_data)
        minimal=10**(-20)
        vargarch=np.zeros(t)

        #CHEATS!
        #Seed the variance with the whole-period variance
        #In practice we would have to have a holdout sample
        #at the beginning and roll the estimate forward.
        vargarch[0]=np.var(array_data)

        #Another cheat: take the mean over the whole period
        #and center the series on that. Hopefully the mean
        #is close to zero. Again in practice to avoid lookahead
        #we would have to roll the mean forward, using only
        #past data.
        overallmean=np.mean(array_data)
        #Compute GARCH(1,1) var's from data given parameters
        for i in range(1,t):
            #Note offset - i-1 observation of data
            #is used for i estimate of variance
            vargarch[i]=c+b*vargarch[i-1]+\
            a*(array_data[i-1]-overallmean)**2
            if vargarch[i]<=0:
                vargarch[i]=minimal
                
        #sum logs of variances
        logsum=np.sum(np.log(vargarch))
        #sum yi^2/sigma^2
        othersum=0
        for i in range(t):
            othersum+=((array_data[i]-overallmean)**2)/vargarch[i]
        #Actually -2 times (6.42) since we are minimizing
        return(logsum+othersum)
    #End of GarchMaxLike

    #Transform parameters to the form used in GarchMaxLike
    #This ensures parameters are in bounds 0<a,b<1, 0<c
    aparam=np.log(initparams[0]/(.999-initparams[0]))
    bparam=np.log(initparams[1]/(.999-initparams[0]-initparams[1]))
    cparam=np.log(initparams[2])
    xinit=[aparam,bparam,cparam]
    #Run the minimization. Constraints are built-in.
    results = scpo.minimize(GarchMaxLike,
                            xinit,
                            method='CG')
    aparam,bparam,cparam=results.x
    a=.999*np.exp(aparam)/(1+np.exp(aparam))
    b=(.999-a)*np.exp(bparam)/(1+np.exp(bparam))
    c=np.exp(cparam)

    return([a,b,c])
    
# %%                 ===================
#                    Correlation Session
#                    ===================
# %% 
#Generate graph of either simulated or historical sample correlation from df_logs
def make_corr_plot(df_logs, rtrial, samplesize, title_str, simulate):
    #Generate a multivariate normal distribution using the data in df_logs
    #compute sample correlations of size samplesize and graph them
    #simulate: False, use historical data in df_logs
    #          True, use simulated data in rtrial
    nobs=len(df_logs)
    periodicity=52
    samplecorrs=[]
    corr_matrix=df_logs[df_logs.columns].corr()
    #Get sample correlations
    if simulate:
        for i in range(samplesize,nobs+1):
            samplecorrs.append(np.corrcoef(rtrial[i-samplesize:i].transpose()))
    else:
        for i in range(samplesize,nobs+1):
            samplecorrs.append(df_logs.iloc[i-samplesize:i] \
                    [df_logs.columns].corr().values)
    sccol=['r','g','b']
    stride=int((nobs-periodicity+1)/(4*periodicity))*periodicity
    dates=df_logs.index[samplesize-1:]
    plot_corrs(dates,samplecorrs,corr_matrix,sccol,stride, \
        title_str+str(samplesize)+'-week sample correlations')
    
def plot_corrs(dates,corrs,corr_matrix,sccol,stride,title_str):
    #dates and corrs have same length
    #dates in datetime format
    #corrs is a list of correlation matrices
    #corr_matrix has the target correlations
    #names of securities are the column names of corr_matrix
    #sccol is colors for lines
    #stride is how many dates to skip between ticks on x-axis
    #title_str is title string
    nobs=len(corrs)
    nsecs=len(corrs[0])
    #plot correlations in corrs, nsec per time period
    ncorrs=nsecs*(nsecs-1)/2
    z=0
    #Go through each pair
    for j in range(nsecs-1):
        for k in range(j+1,nsecs):
            #form time series of sample correlations
            #for this pair of securities
            cs=[corrs[i][j,k] for i in range(nobs)]
            plt.plot(range(nobs),cs, \
                     label=corr_matrix.columns[j]+'/'+ \
                     corr_matrix.columns[k], \
                     color=sccol[z])
            #Show target correlation in same color
            line=[corr_matrix.iloc[j,k]]*(nobs)
            plt.plot(range(nobs),line,color=sccol[z])
            z+=1
    plt.legend()
    tix=[x.strftime("%Y-%m-%d") for x in dates[0:nobs+1:stride]]
    plt.xticks(range(0,nobs+1,stride),tix,rotation=45)
    plt.title(title_str)
    plt.grid()
    plt.show()

# Implementation of Dynamic Conditional Correlations
def deGarch(df_logs):
    '''Garch(1,1) model and deGarch'''
    periodicity=52
    corr_matrix=df_logs[df_logs.columns[1:]].corr()
    overallmean=np.mean(df_logs)
    overallstd=np.std(df_logs)
    tickerlist=df_logs.columns[1:]   #skip the date column
    #Get GARCH params for each ticker
    gparams=[]
    initparams=[.12,.85,.6]
    stgs=[] #Save the running garch sigma's
    for it,ticker in enumerate(tickerlist):
        #Note ORDER MATTERS: make sure values are in date order
        gparams.append(Garch11Fit(initparams, \
            df_logs.sort_values(by="Date")[ticker]))
        a,b,c=gparams[it]
        
        #Create time series of sigmas
        t=len(df_logs[ticker])
        minimal=10**(-20)
        stdgarch=np.zeros(t)
        stdgarch[0]=overallstd[ticker]
        #Compute GARCH(1,1) stddev's from data given parameters
        for i in range(1,t):
            #Note offset - i-1 observation of data
            #is used for i estimate of std deviation
            previous=stdgarch[i-1]**2
            var=c+b*previous+\
                a*(df_logs.sort_values(by="Date")[ticker][i-1] \
                -overallmean[ticker])**2
            stdgarch[i]=np.sqrt(var)
        #Save for later de-GARCHing
        stgs.append(stdgarch)
    #Demeaned, DeGARCHed series go in dfeps
    dfeps=df_logs.sort_values(by="Date").copy()
    for it,ticker in enumerate(tickerlist):
        dfeps[ticker]-=overallmean[ticker]
        for i in range(len(dfeps)):
            dfeps[ticker].iloc[i]/=stgs[it][i]
        print('-'*20)
        print(ticker)
        print('DeGARCHed Mean:',np.mean(dfeps[ticker]))
        print('Raw annualized Std Dev:',np.sqrt(periodicity)*overallstd[ticker])
        print('DeGARCHed Std Dev:',np.std(dfeps[ticker]))
        print('Raw excess kurtosis:',spst.kurtosis(df_logs[ticker]))
        print('DeGARCHed Excess Kurtosis:',spst.kurtosis(dfeps[ticker]))    
    InData=np.array(dfeps[tickerlist])
    return InData
    
def IntegratedCorrObj(s):
    '''Optimize Object for Integrated Correlation'''
    #Compute time series of quasi-correlation
    #matrices from InData using integrated parameter
    #xlam=exp(s)/(1+exp(s)); note this format removes
    #the need to enforce bounds of xlam being between
    #0 and 1. This is applied to formula 9.44.
    #Standardize Q's and apply formula 9.49.
    #Returns scalar 9.49
    xlam=np.exp(s)
    xlam/=1+xlam
    obj9p39=0.
    previousq=np.identity(len(InData[0]))
    #Form new shock matrix
    for i in range(len(InData)):
        #standardize previous q matrix
        #and compute contribution to objective
        #function
        stdmtrx=np.diag([1/np.sqrt(previousq[s,s]) for s in range(len(previousq))])
        previousr=np.matmul(stdmtrx,np.matmul(previousq,stdmtrx))
        #objective function
        obj9p39+=np.log(np.linalg.det(previousr))
        shockvec=np.array(InData[i])
        vec1=np.matmul(shockvec,np.linalg.inv(previousr))
        #This makes obj9p39 into a 1,1 matrix
        obj9p39+=np.matmul(vec1,shockvec)
              
        #Update q matrix
        shockvec=np.mat(shockvec)
        shockmat=np.matmul(shockvec.T,shockvec)
        previousq=xlam*shockmat+(1-xlam)*previousq
    return(obj9p39[0,0])    

def MeanRevCorrObj(params):
    '''Optimize Object for Mean Reverting Correlation'''
    #Compute time series of quasi-correlation
    #matrices from InData using mean reverting
    #formula 9.45. Standardize them and apply
    #formula 9.49. Returns scalar 9.49
    #Extract parameters
    alpha,beta=params
    #Enforce bounds
    if alpha<0 or beta<0:
        return(10**20)
    elif (alpha+beta)>.999:
        return(10**20)
    obj9p39=0
    #Initial omega is obtained through correlation targeting
    Rlong=np.corrcoef(InData.T)
    previousq=np.identity(len(InData[0]))
    #Form new shock matrix
    for i in range(len(InData)):
        #standardize previous q matrix
        #and compute contribution to objective
        #function
        stdmtrx=np.diag([1/np.sqrt(previousq[s,s]) \
                         for s in range(len(previousq))])
        previousr=np.matmul(stdmtrx,np.matmul(previousq,stdmtrx))
        #objective function
        obj9p39+=np.log(np.linalg.det(previousr))
        shockvec=np.array(InData[i])
        vec1=np.matmul(shockvec,np.linalg.inv(previousr))
        #This makes obj9p39 into a 1,1 matrix
        obj9p39+=np.matmul(vec1,shockvec)
              
        #Update q matrix
        shockvec=np.mat(shockvec)
        shockmat=np.matmul(shockvec.T,shockvec)
        previousq=(1-alpha-beta)*Rlong+alpha*shockmat+beta*previousq
    return(obj9p39[0,0])
    
def compute_integratedCorr(df_logs,InData):
    '''Compute integrated correlations'''
    periodicity = 52
    result=minimize_scalar(IntegratedCorrObj)
    xlamopt=np.exp(result.x)
    xlamopt/=1+xlamopt
    print('Optimal lambda:',xlamopt)
    print('Optimal objective function:', \
          result.fun)
    if xlamopt>=1 or xlamopt==0:
        halflife=0
    else:
        halflife=-np.log(2)/np.log(1-xlamopt)
    print('Half-life (years):',halflife/periodicity)
    
    #Compute integrated correlations
    nobs=len(InData)
    nsecs=len(InData[0])
    #Start quasi-correlation matrix series with identity
    previousq=np.identity(nsecs)
    rmatrices=[]
    for i in range(nobs):
        stdmtrx=np.diag([1/np.sqrt(previousq[s,s]) for s in range(nsecs)])
        rmatrices.append(np.matmul(stdmtrx,np.matmul(previousq,stdmtrx)))
        shockvec=np.mat(np.array(InData[i]))
        #Update q matrix
        shockmat=np.matmul(shockvec.T,shockvec)
        previousq=xlamopt*shockmat+(1-xlamopt)*previousq
    
    #Plot integrated correlations
    iccol=['r','g','b']
    xtitle='Integrated correlations λ=%1.5f' % xlamopt
    xtitle+=', '+min(df_logs.index.strftime("%Y-%m-%d"))+':'+ \
                      max(df_logs.index.strftime("%Y-%m-%d"))
    dates=df_logs.index
    stride=5*periodicity
    corr_matrix=df_logs[df_logs.columns].corr()
    plot_corrs(dates,rmatrices,corr_matrix,iccol,stride,xtitle)
    
def compute_meanRevCorr(df_logs,InData):
    '''Compute mean reverting correlations'''
    periodicity = 52
    #alpha and beta positive
    corr_bounds = scpo.Bounds([0,0],[np.inf,np.inf])
    #Sum of alpha and beta is less than 1
    corr_linear_constraint = \
        scpo.LinearConstraint([[1, 1]],[0],[.999])
    
    initparams=[.02,.93]
    
    results = scpo.minimize(MeanRevCorrObj, \
            initparams, \
            method='trust-constr', \
            jac='2-point', \
            hess=scpo.SR1(), \
            bounds=corr_bounds, \
            constraints=corr_linear_constraint)
    
    alpha,beta=results.x
    print('Optimal alpha, beta:',alpha,beta)
    print('Optimal objective function:',results.fun)
    halflife=-np.log(2)/np.log(1-alpha)
    print('Half-life (years):',halflife/periodicity)
    
    #Compute mean reverting correlations
    nobs=len(InData)
    nsecs=len(InData[0])
    previousq=np.identity(nsecs)
    Rlong=np.corrcoef(InData.T)
    rmatrices=[]
    for i in range(nobs):
        stdmtrx=np.diag([1/np.sqrt(previousq[s,s]) for s in range(nsecs)])
        rmatrices.append(np.matmul(stdmtrx,np.matmul(previousq,stdmtrx)))
        shockvec=np.mat(np.array(InData[i]))
        #Update q matrix
        shockmat=np.matmul(shockvec.T,shockvec)
        previousq=(1-alpha-beta)*Rlong+alpha*shockmat+beta*previousq
    
    #Plot mean-reverting correlations
    iccol=['r','g','b']
    xtitle='Mean Reverting Correlations α=%1.5f' % alpha
    xtitle+=', β=%1.5f' % beta
    xtitle+=', '+min(df_logs.index.strftime("%Y-%m-%d"))+':'+ \
                 max(df_logs.index.strftime("%Y-%m-%d"))
    dates=df_logs.index
    stride=5*periodicity
    corr_matrix=df_logs[df_logs.columns].corr()
    plot_corrs(dates,rmatrices,corr_matrix,iccol,stride,xtitle)

def MacGyver_method(tickerlist,dfeps,method='integrated'):
    '''MacGyver method of pairwise calculation'''
    periodicity = 52
    minimal=10**(-20)
    xlams,alphas,betas=[],[],[]
    for it in range(len(tickerlist)-1):
        tick1=tickerlist[it]
        for jt in range(it+1,len(tickerlist)):
            tick2=tickerlist[jt]
            InData=np.array(dfeps[[tick1,tick2]]) # global cast
            # integrated corr.
            if method == 'integrated':
                result=minimize_scalar(IntegratedCorrObj)
                xlamopt=np.exp(result.x)/(1+np.exp(result.x))
                print(tick1,tick2)
                print('    Optimal lambda:',xlamopt)
                print('    Optimal objective function:', \
                      result.fun)
                if np.absolute(xlamopt)<minimal or xlamopt>=1:
                    halflife=0
                else:
                    halflife=-np.log(2)/np.log(1-xlamopt)
                print('    Half-life (years):',halflife/periodicity)
                xlams.append(xlamopt)
            # mean revert corr.
            elif method == 'meanRev':
                #alpha and beta positive
                corr_bounds = scpo.Bounds([0,0],[np.inf,np.inf])
                #Sum of alpha and beta is less than 1
                corr_linear_constraint = \
                    scpo.LinearConstraint([[1, 1]],[0],[.999])
            
                initparams=[.02,.93]
                results = scpo.minimize(MeanRevCorrObj, \
                        initparams, \
                        method='trust-constr', \
                        jac='2-point', \
                        hess=scpo.SR1(), \
                        bounds=corr_bounds, \
                        constraints=corr_linear_constraint)
                alpha,beta=results.x
                print('Optimal alpha, beta:',alpha,beta)
                print('Optimal objective function:',results.fun)
                halflife=-np.log(2)/np.log(1-alpha)
                print('Half-life (years):',halflife/periodicity)
                alphas.append(alpha)
                betas.append(beta)
     # cout median values
    if method == 'integrated':
        print('\nMedian MacGyver lambda:',np.median(xlams))
    elif method == 'meanRev':
        print('\nMedian MacGyver alpha:',np.median(alphas))
        print('\nMedian MacGyver beta:',np.median(betas))

