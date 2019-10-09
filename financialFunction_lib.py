 # -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 23:58:47 2017
Financial Functions Lib
@author: Roy
可使用包ffn里内置函数代替
"""


def downsideDeviation(se_return,MARR):
    '''
    下行偏差计算函数 描述下行风险
    '''
    import numpy as np
    se_return = np.asarray(se_return)
    temp = se_return[se_return<MARR]#小于可接受最低收益率的序列Minimum Acceptable Rate of Return
    downDevi = (sum((MARR-temp)**2)/len(se_return))**0.5
    return downDevi
#-----------------------------------------------------------------------------
def valueAtRisk(se_return,alpha=0.05):
    '''
    风险价值VaR
    '''
    from scipy.stats import norm
    value1 = se_return.quantile(alpha)#历史模拟法
    value2 = norm.ppf(alpha,se_return.mean(),se_return.std()) #协方差矩阵法
    return [value1,value2]
#-----------------------------------------------------------------------------
def expectedShortfall(se_return,alpha=0.05):
    '''
    期望亏空ES
    '''
    from scipy.stats import norm
    value1 = se_return[(se_return<=se_return.quantile(alpha))].mean()#历史模拟法
    value2 = se_return[(se_return<=norm.ppf(alpha,se_return.mean(),se_return.std()))].mean()#协方差矩阵法
    return [value1,value2] 
#-----------------------------------------------------------------------------
def momentum(se_price,period):
    '''
    动量计算
    '''
    se_lagPrice = se_price.shift(period)
    momen = se_price-se_lagPrice #价差作为动量
    momen1 = float(se_price)/se_lagPrice-1 #百分比变化作为动量
    momen = momen.dropna()    
    momen1 = momen1.dropna()
    return [momen,momen1]
#-----------------------------------------------------------------------------
def rsi(price,period=6):
    '''
    相对弱强指数计算
    (自动去除na)
    '''
    import pandas as pd
    import numpy as np
    clprcChange=price-price.shift(1)
    clprcChange=clprcChange.dropna()
    indexprc=clprcChange.index
    upPrc=pd.Series(0,index=indexprc)
    upPrc[clprcChange>0]=clprcChange[clprcChange>0]
    downPrc=pd.Series(0,index=indexprc)
    downPrc[clprcChange<0]=-clprcChange[clprcChange<0]
    rsidata=pd.concat([price,clprcChange,upPrc,downPrc],axis=1)
    rsidata.columns=['price','PrcChange','upPrc','downPrc']
    rsidata=rsidata.dropna();
    SMUP=[]
    SMDOWN=[]
    for i in range(period,len(upPrc)+1):
        SMUP.append(np.mean(upPrc.values[(i-period):i],dtype=np.float32))
        SMDOWN.append(np.mean(downPrc.values[(i-period):i],dtype=np.float32))
        rsi=[100*SMUP[i]/(SMUP[i]+SMDOWN[i]) for i in range(0,len(SMUP))]
    indexRsi=indexprc[(period-1):]
    rsi=pd.Series(rsi,index=indexRsi)
    return(rsi)
#-----------------------------------------------------------------------------
def sma(se_Price,k):
    '''简单移动平均'''
    import pandas as pd
    Sma=pd.Series(0.0,index=se_Price.index)
    for i in range(k-1,len(se_Price)):
        Sma[i]=sum(se_Price[(i-k+1):(i+1)])/k
    return(Sma)

def wma(se_Price,weight):
    '''加权移动平均'''
    import pandas as pd
    import numpy as np
    k=len(weight)
    arrWeight=np.array(weight)
    Wma=pd.Series(0.0,index=se_Price.index)
    for i in range(k-1,len(se_Price.index)):
        Wma[i]=sum(arrWeight*se_Price[(i-k+1):(i+1)])
    return(Wma)

def ewma(se_Price,period=5,exponential=0.2):
    '''指数加权移动平均'''
   import pandas as pd
   Ewma=pd.Series(0.0,index=se_Price.index)
   Ewma[period-1]=np.mean(se_Price[0:period])
   for i in range(period,len(se_Price)):
       Ewma[i]=exponential*se_Price[i]+(1-exponential)*Ewma[i-1]
   return(Ewma)   
#-----------------------------------------------------------------------------
def bollingBands(se_Price,period=20,times=2):
    '''
    布林带计算
    '''
    upBBand=pd.Series(0.0,index=se_Price.index)
    midBBand=pd.Series(0.0,index=se_Price.index)
    downBBand=pd.Series(0.0,index=se_Price.index)
    sigma=pd.Series(0.0,index=se_Price.index)
    for i in range(period-1,len(se_Price)):
        midBBand[i]=np.nanmean(se_Price[i-(period-1):(i+1)])
        sigma[i]=np.nanstd(se_Price[i-(period-1):(i+1)])
        upBBand[i]=midBBand[i]+times*sigma[i]
        downBBand[i]=midBBand[i]-times*sigma[i]
    BBands=pd.DataFrame({'upBBand':upBBand[(period-1):],\
                         'midBBand':midBBand[(period-1):],\
                         'downBBand':downBBand[(period-1):],\
                         'sigma':sigma[(period-1):]})
    return(BBands)
    
def bollingRisk(se_Price,multiplier):
    '''
    布林带风险计算(基于统计学正态分布)
    '''
    k=len(multiplier)
    overUp=[]
    belowDown=[]
    BollRisk=[]
    for i in range(k):
        BBands=bbands(se_Price,20,multiplier[i])
        a=0
        b=0
        for j in range(len(BBands)):
            se_Price=se_Price[-(len(BBands)):]
            if se_Price[j]>BBands.upBBand[j]:
                a+=1
            elif se_Price[j]<BBands.downBBand[j]:
                b+=1
        overUp.append(a)
        belowDown.append(b)
        BollRisk.append(100*(a+b)/len(se_Price))
    return(BollRisk)
#-----------------------------------------------------------------------------
#RESTORE
def get_GSISI(array_return,array_beta,alpha=0.05):
    '''
    投资者情绪指数(申万一级市场行业CAPM)
    '''
    import scipy.stats as sta
    result = sta.spearmanr(array_return,array_beta)
    corr = result[0]
    pvalue = result[1]
    if (pvalue <= alpha): #结果显著
        goodness = 1
    else: goodness = 0
    return [100*corr,goodness]
        
    
    






