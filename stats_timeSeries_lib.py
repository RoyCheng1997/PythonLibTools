# -*- coding: utf-8 -*-
"""
Created on Tue May 29 21:01:58 2018
金融计量的python实现
arch模型只兼容64位
@author: Roy
"""
from scipy import stats
from statsmodels.tsa import stattools
import statsmodels.api as st
from statsmodels.graphics import tsaplots 
import numpy as np
from arch import arch_model
from arch.unitroot import ADF


# ================================================================
# 1. 统计性描述
#=================================================================
array = np.asarray([1,2,3,4,5])
array_1 = np.asarray([1,2,3,4,5])
array_y = np.asarray([1,2,3,4,5])
'''样本方差'''
std = np.std(array)
'''样本偏度'''
skewness = stats.skew(array)
'''样本峰度'''
kurtosis = stats.kurtosis(array)
'''样本均值的标准误(总体标准差未知用以替代标准差scale)'''
sem = stats.sem(array) 

'''自相关系数序列 0-nlags'''
arr_acf = stattools.acf(array,nlags=40,unbiased=False,qstat=False,alpha=None) #自相关系数autocorrelation coefficient function
#unbiased 是否调整分母使结果无偏 nlags 设置最大滞后期数 qstat 是否返回Ljung-Box(白噪声检验)的结果 alpha是否计算置信区间
'''偏自相关系数 0-nlags'''
arr_pacf = stattools.pacf(array,nlags=40,method='ywunbaised',alpha = None) #偏自相关系数 partial autocorrelation coefficient function
'''自相关函数图'''
tsaplots.plot_acf(array,use_vlines=True,lags=40) #各阶自相关系数图 两直线间的自相关系数不显著异于0
tsaplots.plot_pacf(array,use_vlines=True,lags=40)#各阶偏自相关系数图 两直线间的偏自相关系数不显著异于0


# ================================================================
# 2. 检验
#=================================================================
'''正态性检验'''
normTest = stats.normaltest(array,nan_policy='omit') #not norm, reject norm
pValue = normTest.pvalue
'''区间估计(置信水平，自由度，样本均值，标准差) t分布'''
stats.t.interval(0.95,len(array)-1,np.mean(array),stats.sem(array))
'''单总体t检验(H0 x=5) 若p<0.05则拒绝原假设'''
stats.ttest_1samp(array,5)
'''双独立总体t检验(H0 x=y) 若p<0.05则拒绝原假设'''
stats.ttest_ind(array,array_1)
'''配对样本t检验(H0 x-y=0) 若p<0.05则拒绝原假设 两样本不一定独立'''
stats.ttest_rel(array,array_1)
'''Durbin-Watson检验自相关性'''
st.stats.durbin_watson(array) # array为residuals
'''ADF平稳性检验'''
stattools.adfuller(array, maxlag=None, regression='c', autolag='AIC', store=False, regresults=False)
#return
    #adf : float Test statistic
    #pvalue : float MacKinnon’s approximate p-value based on MacKinnon (1994, 2010)
    #usedlag : int Number of lags used
    #nobs : int Number of observations used for the ADF regression and calculation of the critical values
    #critical values : dict Critical values for the test statistic at the 1 %, 5 %, and 10 % levels. Based on MacKinnon (2010)
    #icbest : float The maximized information criterion if autolag is not None.
    #resstore : ResultStore, optional  A dummy class with results attached as attributes
'''EG协整检验'''
stattools.coint(array, array_1, trend='c', method='aeg', maxlag=None, autolag='aic', return_results=None)
#return
    #coint_t : float
    #t-statistic of unit-root test on residuals
    #pvalue : float
    #MacKinnon’s approximate, asymptotic p-value based on MacKinnon (1994)
    #crit_value : dict
    #Critical values for the test statistic at the 1 %, 5 %, and 10 % levels based on regression curve. This depends on the number of observations
'''Ljung-Box白噪声检验'''
x = stattools.acf(array,nlags=40,unbiased=False,qstat=False,alpha=None)
nobs = array
stattools.q_stat(x,nobs,type='ljungbox') 
# x所检验的自相关系数序列 nobs计算自相关系数序列x所用的样本数n
# return
    #检验的统计量array 
    #p值的array 若p小于0.05 则拒绝原假设 则不是白噪声 存在自相关性    
    
# ================================================================
# 3. 回归
#=================================================================
# OLS ----------------------------
'''OLS'''
array1 = st.add_constant(array)
model = st.OLS(array_y,array1).fit()
print(model.summary())
print(model.fittedvalues) # fit values
print(model.bic) # bic
print(model.pvalues) #pvalue
print(model.rsquared) #r^2

# 2SLS ----------------------------
def two_sls(dependt_arr,exogList,endog_arr,instruments_arr):
    '''自定义2sls算法'''
    #内生变量对工具变量的回归
    model1 = st.OLS(endog_arr,instruments_arr).fit()
    #yhat
    hat = model1.predict(instruments_arr)
    #添加yhat到回归中
    exogList.append(list(hat))
    exog = np.asarray(exogList).T
    #用yhat代替y回归
    model2 = st.OLS(dependt_arr,exog).fit()
    print(model2.summary())
    return model2.params

# ================================================================
# 3. 时间序列模型
#=================================================================
"""ARIMA模型"""    
#step1 检查是否是平稳序列(单位根检验)
#array为训练数据集 最大滞后阶数过多导致pvalue过低 若p小于显著性水平 则拒绝原假设 认为该序列是平稳的 进行下一步
print (ADF(array,max_lags=10)).summary().as_text()
#step2 检查是否是白噪声
LjungBox = stattools.q_stat(stattools.acf(array)[1:12],len(array))
pvalue = LjungBox[1][-1] #pvalue小于显著性水平 则拒绝原假设 不是白噪声 进行下一步
#step3 识别模型参数p和q
#建立低阶p q的各种组合情况下的ARMA模型 并运用AIC准则进行比较 选出AIC值最小的模型
model1 = arima_model.ARIMA(se_train.order=(1,0,1)).fit() #(p,q)=(1,1)
model1.aic()#model1.summary()
model2 = arima_model.ARIMA(se_train.order=(1,0,2)).fit() #(p,q)=(1,2)
model2.aic()#model2.summary()
model3 = arima_model.ARIMA(se_train.order=(2,0,1)).fit() #(p,q)=(2,1)
model3.aic()#model3.summary()
model4 = arima_model.ARIMA(se_train.order=(2,0,2)).fit() #(p,q)=(2,2)
model4.aic()#model4.summary()
model5 = arima_model.ARIMA(se_train.order=(3,0,1)).fit() #(p,q)=(3,1)
model5.aic()#model5.summary()
model6 = arima_model.ARIMA(se_train.order=(3,0,2)).fit() #(p,q)=(3,2)
model6.aic()#model6.summary()
#step4 检验模型中系数的显著性
model6.conf_int() #返回系数的置信区间
#step5 模型诊断 
#对残差进行检验 确保其服从正态分布的白噪声序列
stdresid = model6.resid/math.sqrt(model6.sigma2) 
LjungBox = stattools.q_stat(stattools.acf(stdresid)[1:13],len(stdresid)) #增加lag可以减小p
pvalue = LjungBox[1][-1]
#运用模型进行预测
model6.forcast(3)[0] #预测后三个数据


'''GARCH模型'''#条件方差即波动率
am = arch_model(array) #默认建立GARCH(1,1)模型
model = am.fit(update_freq=0) #表示不输出中间结果 只输出最终结果
print(model.summary())