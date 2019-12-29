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
from scipy.optimize import minimize,fsolve
from math import e 
import scipy.stats as spst
from scipy.stats import norm
import numpy.random      as ran     # random number generators
import copy
import random
# %%
class Option(object):
    '''
    Option Object for pricing and related calculation
    Note: All rates denote as true value
    
    Parameters
    ----------
                
    Methods
    -------

    
    Examples
    --------
    >>> x = Option(S=100,K=110,sigma=.15,r=.05,d=0,t=1)
    >>> x.price_BlackScholes(dic)
    >>> x.price_simulation()

    '''    
    
    def __init__(self,S,K,sigma,r,d,t,type="European",direction="Call",dic={}):
        self.s = S            # spot price
        self.k = K            # strike price
        self.sigma = sigma    # volatility
        self.r = r            # risk-free/funding rate
        self.d = d          # dividend rate
        self.t = t            # maturity
        self.type = type      # type: European/American/Exotic
        self.direction = direction # Call/Put
        self.dic = dic        # further parameters for Exotic options
        
        self.price = 0
        self.impliedVol = 0
        
        
    def price_BlackScholes(self,vol=-1):
        '''Use Black Scholes Model to calculate European Options Price'''
        # type detection
        if self.type != 'European':
            print('Option Type NOT Supported')
            return
        # default setting, also used to calculate implied vol
        if vol == -1:
            vol = self.sigma
        else:
            pass
        # calculation
        d1 = (np.log(self.s/self.k) + (self.r-self.d+self.sigma**2/2)*self.t)/self.sigma/np.sqrt(self.t)
        d2 = d1 - self.sigma * np.sqrt(self.t)
        if self.direction == 'Call':
            c = self.s*np.exp(-self.d*self.t) * norm.cdf(d1) - self.k*np.exp(-self.r*self.t)*norm.cdf(d2)
            self.price = c
        elif self.direction == 'Put':
            p = self.k*np.exp(-self.r*self.t)*norm.cdf(-d2) - self.s*np.exp(-self.d*self.t) * norm.cdf(-d1)
            self.price = p
        else:
            raise IndexError('Direction must be Call or Put')
        return self.price
            
    def price_simulation(self,movement='GeometricBM',payoff=None,simulationN=10000,steps=1000):
        '''Use Simulation to price options'''
        if self.d != 0:
            print('Warning: In simulation, we set dividend rate equals to 0')
        else:
            pass
        
        dt = self.t/steps
        priceList = []
        for i in range(simulationN):
            # generate path
            path = [self.s]
            for j in range(steps):
                st = path[-1]
                if movement == 'GeometricBM':
                    ds = self.r*st*dt + self.sigma*st*ran.normal(0,np.sqrt(dt))
                elif movement == 'BM':
                    ds = self.r*self.s*dt + self.sigma*self.s*ran.normal(0,np.sqrt(dt))
                else:
                    raise IndexError("Movement type is not supported")
                path.append(st+ds)
            # calculate price
            if payoff is not None:
                price = payoff(path) # payoff is a function
            elif self.type == 'European':
                if self.direction == 'Call': # call
                    price = max(path[-1]-self.k,0) * np.exp(-self.r*self.t)
                else: # put
                    price = max(self.k-path[-1],0) * np.exp(-self.r*self.t)                   
            elif self.type == 'American':
                pass 
            else:
                pass
            
            priceList.append(price)
        return np.mean(priceList)
    
    
    def price_biTree():
        pass
    
    def price_triTree():
        pass
    
    def price_exotic():
        pass
           
    def calculate_impliedVol(method='BS'):
        pass
    
    
    def calculate_impliedData(dic={}):
        pass
    
    def calculate_greeks():
        pass
    



# %%
class Bonds(object):
    '''
    Bonds Object for pricing and related calculation
    Note: All rates denote as %
    
    Parameters
    ----------
    _couponRate: float % 
                Coupon paying rate for simple coupon bonds (the field will be invalid if dict data is input)
    _maturity: int
                Maturity of the bond, only int is supported (the field will be invalid if dict data is input)
    _fundingRate: float % 
                Funding/Discounting rate for simple coupon bonds (the field will be invalid if dict data is input)       
    _dict: dict
                Input data to calculate everything
    _compound: str
                Compound method for input data
    _payTerm: int
                Total paying times for input data
    price/duration/convexity/yieldRate: as name disclosed    
                
    Methods
    -------
    calculate_price
    calculate_yield
    calculate_convexity
    calculate_yield
        
    price_simple
    duration_simple
    convexity_simple
        
    plot:
        
    cvx_adjust:
    
    Examples
    --------
    >>> x = Bonds()
    >>> cashFlow,time,fundingRate = [4,5,6,7,8,9,110],[1,2,3,4,5,6,7],[1,1.5,2,2.5,3,3.5,4]
    >>> dic={'cashFlow':cashFlow,'time':time,'fundingRate':fundingRate}
    >>> x.calculate_price(dic)
    >>> x.calculate_yield()
    >>> x.calculate_duration()
    >>> x.calculate_convexity()

    '''
    def __init__(self,couponRate=3,maturity=10,fundingRate=2):
        '''constructor'''
        self._maturity = maturity
        self._couponRate = couponRate
        self._fundingRate = fundingRate   
        self._dict = {} # input data
        self._compound = ''
        self._payTerm = 0
        
        self.price = 0
        self.duration = 0
        self.convexity = 0
        self.yieldRate = couponRate # for coupon bond, default yield = coupon rate
        
    def calculate_price(self,dic={'cashFlow':[],'time':[],'fundingRate':[]},compound='continous'):
        '''
        General price by input dict, since bonds are trivial, so out data solves everything
        @ author: RoyCheng
        
        Parameters
        ----------
        dic: dict, input data, please follow the form
        compound: str, compound method (continous/annual/semi-annual)
        
        Returns
        -------
        price: float, price of the coupon bond
        '''
        self._dict = dic # global field
        self._compound = compound  # global field
        self._payTerm = len(self._dict['cashFlow'])
        
        cf = np.asarray(dic['cashFlow'])
        time = np.asarray(dic['time'])
        funding = np.asarray(dic['fundingRate'])
        if compound == 'continous':
            discount = np.exp(-funding*time/100)
        elif compound == 'annual':
            discount = 1/((1+funding/100)**time)
        elif compound == 'semi-annual':
            discount = 1/((1+funding/100/2)**(2*time))
        else:
            raise IndexError("Compound method is not available")
        self.price = np.sum(cf*discount) # global field
        return self.price
        
    def calculate_yield(self):
        '''Calculate Yield Rate'''
        y0 = np.mean(self._dict['fundingRate'])
        result = fsolve(self.__calYield__,y0)
        self.yieldRate = result[0]
        return self.yieldRate
    
    def calculate_duration(self,dy=0.01,type='Macauley'):
        '''Calculate Duration'''
        if self.yieldRate == 3: # default value, not calculated yield
            self.calculate_yield()
        else:
            pass
        dp_dy = (self.__calPrice__(self.yieldRate + dy) - self.__calPrice__(self.yieldRate - dy))/2/dy*100
        if type == 'Macauley':
            self.duration = -dp_dy/self.price*(1+self.yieldRate/100)
        elif type == 'Modified':
            self.duration = -dp_dy/self.price
        else:
            raise ValueError("Type not in ['Macauley','Modified']")
        return self.duration
    
    def calculate_convexity(self,dy=0.01):
        '''Calculate Convexity'''
        if self.yieldRate == 3: # default value, not calculated yield
            self.calculate_yield()
        else:
            pass
        dp_dy2 = (self.__calPrice__(self.yieldRate + dy) + self.__calPrice__(self.yieldRate - dy) - 2* self.price)/((dy*100)**2)
        self.convexity = dp_dy2/self.price
        return self.convexity
        
    def price_simple(self):
        '''
        price simple coupon bond
        Pt = sum(cy^i)+100y^T       
        [fixed coupon/funding rate, integer maturity, faceValue=100, annual pay]
        @ author: Prof. Kenneth Winston
        
        Parameters
        ----------
        Returns
        -------
        price: float, price of the coupon bond
        '''
        #with annual coupon c, t years to
        #maturity, discount rate r
        c,r,t = self._couponRate*100,self._fundingRate,self._maturity
        
        if r<=-100:  #Unreasonable discount rate
            return(100)
        y=1/(1+r/100)
        price=100*(y**t)
        if (y==1):   #no discount rate
            geometric=t
        else:
            geometric=(1-y**t)/(1-y)
        price += geometric*c*y
        self.price = price
        return(price)
    
    def duration_simple(self,type='Macauley'):
        '''
        calculate duration of simple coupon bond
        [fixed coupon/funding rate, integer maturity, faceValue=100, annual pay]
        @ author: Prof. Kenneth Winston/RoyCheng
        
        Parameters
        ----------
        type: string(Macauley/Modified)
        
        Returns
        -------
        duration: float, Macauley duration of the coupon bond
        '''
        c,r,t = self._couponRate*100,self._fundingRate,self._maturity
        if r<=-100:  #Unreasonable discount rate
            return(0)
        y=1/(1+r/100)
        ytothet= y**t
        duration = 100*t*ytothet
        if (y==1):   #no discount rate
            multiplier=t*(t+1)/2
        else:
            multiplier=(1-ytothet-t*(1-y)*ytothet)/(1-y)**2
        duration += multiplier*c*y
        price = self.price_simple()   #Rescale by price
        duration/=price
        if type == 'Macauley':
            duration = duration
        elif type == 'Modified':
            duration *= y
        else:
            raise ValueError("Type not in ['Macauley','Modified']")
        self.duration = duration
        return(duration)
    
    def convexity_simple(self):
        '''
        convexity of simple coupon bond
        [fixed coupon/funding rate, integer maturity, faceValue=100, annual pay]
        @ author: Prof. Kenneth Winston
        
        Parameters
        ----------
        
        Returns
        -------
        convexity: float, convexity of the coupon bond
        '''
        #maturity, discount rate r
        c,r,t = self._couponRate*100,self._fundingRate,self._maturity
        if r<=-100:  #Unreasonable discount rate
            return(0)
        y=1/(1+r/100)
        ytothet=y**t
        convexity=100*t*(t+1)*ytothet*(y**2)
        if (y==1):   #no discount rate
            ytttterm=0
        else:
            ytttterm=-(t+1)*(t+2)+2*t*(t+2)*y-t*(t+1)*y**2
            ytttterm*=ytothet
            ytttterm+=2
            ytttterm*=c*(y/(1-y))**3
        convexity+=ytttterm
        price=self.price_simple()   #Rescale by price
        convexity/=price
        self.convexity = convexity
        return(convexity)
    
    def __calYield__(self,y):
        '''Auxillary func for calculate yield'''
        price = self.__calPrice__(y)
        return price-self.price
    
    def __calPrice__(self,y):
        '''Auxillary func for calculate yield'''
        dic = {'cashFlow':self._dict['cashFlow'],'time':self._dict['time'],'fundingRate':[y]*self._payTerm}
        price = self.calculate_price(dic,self._compound)
        return price
    
