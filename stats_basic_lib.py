# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 23:24:36 2017
Statistics Lib
@author: Roy
"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.stats.anova as anova
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa import stattools
from statsmodels.graphics import tsaplots 
import statsmodels.api as sm
import matplotlib.pyplot as plt
from arch.unitroot import ADF
from statsmodels.tsa import arima_model
from arch import arch_model

#产生随机数=================================================================================
randomList = np.random.choice([1,2,3,4,5],size = 100, replace = True,p=[0.1,0.1,0.3,0.3,0.2]) #离散概率分布

random = np.random.binomial(100,0.5,size=100)#二项分布
stats.binom.pmf(50,100,0.5)#概率质量函数 100次试验50次成功的概率 单点概率
stats.binom.cdf(50,100,0.5)#累积密度函数                        概率累积

random = np.random.normal(loc=0,scale=1,size=100)#正态分布
random = np.random.standard_normal(size=100)#标准正态分布
density = stats.kde.gaussian_kde(Series) #模拟概率密度曲线(计算方差均值再拟合正太)
stats.norm.pdf(random)
stats.norm.cdf(random)

random = np.random.chisquare(5,size=100)#卡方分布
stats.chi.pdf(5,random)
stats.chi.cdf(5,random)

random = np.random.f(5,4,size=100)#F分布
stats.f.pdf(5,4,random)
stats.f.cdf(5,4,random)

#统计推断=================================================================================
x = [10.1,10,9.8,10.5,9.7,10.1,9.9,10.2,10.3,9.9]
y = np.random.chisquare(5,size=10)
stats.sem(x) #样本均值的标准误(总体标准差未知用以替代标准差scale)
stats.t.interval(0.95,len(x)-1,np.mean(x),stats.sem(x))#区间估计(置信水平，自由度，样本均值，标准差) t分布
stats.ttest_1samp(x,10)#单总体t检验(H0 x=10) 若p<0.05则拒绝原假设
stats.ttest_ind(x,y)#双独立总体t检验(H0 x=y) 若p<0.05则拒绝原假设
stats.ttest_rel(x,y)#配对样本t检验(H0 x-y=0) 若p<0.05则拒绝原假设 两样本不一定独立

#ANOVA=================================================================================
year_return = pd.read_csv('dat\\TRD_Year.csv',encoding='gbk') #注意不能再console里直接运行 dat\\需要当前路径 需要直接运行该文档
PSID = pd.read_csv('dat\\PSID.csv')
model1 = OLS.from_formula('Return ~ C(Industry)',data = year_return.dropna()).fit()
table1 = anova.anova_lm(model1) #p=PR 原假设:不同行业对return没有影响 p<a则拒绝原假设
print(table1)
model2 = OLS.from_formula('earnings ~ C(married)+C(educatn)',data = PSID.dropna()).fit()
table2 = anova.anova_lm(model2)
print(table2)                                                             
model3 = OLS.from_formula('earnings ~ C(married)*C(educatn)',data = PSID.dropna()).fit() #增加交互项
table3 = anova.anova_lm(model3)
print(table3)

#回归分析=================================================================================
#一元回归模型
TRD_Index=pd.read_table('dat\\TRD_Index.txt',sep='\t')
SHindex=TRD_Index[TRD_Index.Indexcd==1] 
SZindex=TRD_Index[TRD_Index.Indexcd==399106]
SHRet=SHindex.Retindex
SZRet=SZindex.Retindex
SZRet.index=SHRet.index
model1=sm.OLS(SHRet,sm.add_constant(SZRet)).fit()
print(model1.summary())
fitList = model1.fittedvalues
bic = model1.bic
p_value = model1.pvalues
r2 = model1.rsquared
#残差分析
plt.scatter(model1.fittedvalues,model1.resid) #残差图(0附近随机分布)
plt.xlabel(u'拟合值')
plt.ylabel(u'残差')
sm.qqplot(model1.resid_pearson,stats.norm,line='45') #检验残差的正态性(直线附近)
plt.scatter(model1.fittedvalues,model1.resid_pearson**0.5) #检验残差的同方差性(均匀带状)
plt.xlabel(u'拟合值')
plt.ylabel(u'标准化残差的平方根')
#多元回归模型
penn=pd.read_excel('dat\\Penn World Table.xlsx',2)#读取第二个表
model2=sm.OLS(np.log(penn.rgdpe),sm.add_constant(penn.iloc[:,-6:])).fit()
print(model2.summary())
penn.iloc[:,-6:].corr() #相关性矩阵

#时间序列模型=================================================================================
arr_acf = stattools.acf(x,nlags=40,unbiased=False,qstat=False,alpha=None) #自相关系数autocorrelation coefficient function
#unbiased 是否调整分母使结果无偏 nlags 设置最大滞后期数 qstat 是否返回Ljung-Box的结果 alpha是否计算置信区间
arr_pacf = stattools.pacf(x,nlags=40,method='ywunbaised',alpha = None) #偏自相关系数 partial autocorrelation coefficient function
tsaplots.plot_acf(x,use_vlines=True,lags=30) #各阶自相关系数图 两直线间的自相关系数不显著异于0
tsaplots.plot_pacf(x,use_vlines=True,lags=30)#各阶偏自相关系数图 两直线间的偏自相关系数不显著异于0

stattools.adfuller(x, maxlag=None, regression='c', autolag='AIC', store=False, regresults=False)
'''ADF平稳性检验'''
###return
#adf : float Test statistic
#pvalue : float MacKinnon’s approximate p-value based on MacKinnon (1994, 2010)
#usedlag : int Number of lags used
#nobs : int Number of observations used for the ADF regression and calculation of the critical values
#critical values : dict Critical values for the test statistic at the 1 %, 5 %, and 10 % levels. Based on MacKinnon (2010)
#icbest : float The maximized information criterion if autolag is not None.
#resstore : ResultStore, optional  A dummy class with results attached as attributes
stattools.coint(y0, y1, trend='c', method='aeg', maxlag=None, autolag='aic', return_results=None)
'''EG协整检验'''
###return
#coint_t : float
#t-statistic of unit-root test on residuals
#pvalue : float
#MacKinnon’s approximate, asymptotic p-value based on MacKinnon (1994)
#crit_value : dict
#Critical values for the test statistic at the 1 %, 5 %, and 10 % levels based on regression curve. This depends on the number of observations
stattools.q_stat(x,nobs,type='ljungbox') 
'''Ljung-Box白噪声检验'''
#x所检验的自相关系数序列 nobs计算自相关系数序列x所用的样本数n
#返回检验的统计量array 和p值的array 若p小于0.05 则拒绝原假设 则不是白噪声 存在自相关性
#----------------------------------------------------------------------------
'''ARIMA模型'''
#step1 检查是否是平稳序列
#se_train为训练数据集 最大滞后阶数过多导致pvalue过低 若p小于显著性水平 则拒绝原假设 认为该序列是平稳的 进行下一步
print (ADF(se_train,max_lags=10)).summary().as_text()
#step2 检查是否是白噪声
LjungBox = stattools.q_stat(stattools.acf(se_train)[1:12],len(se_train))
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
#----------------------------------------------------------------------------
'''GARCH模型'''#条件方差即波动率
am = arch_model(series) #默认建立GARCH(1,1)模型
model = am.fit(update_freq=0) #表示不输出中间结果 只输出最终结果
print(model.summary())









