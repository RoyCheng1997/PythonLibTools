# -*- coding: utf-8 -*-
"""
Created on Wed May  3 20:09:56 2017

@author: RoyCheng
"""

import pandas as pd
import numpy as np
import sys
import json
# %%
#Series ================================================================
obj = pd.Series([4,7,-5,3],index=['a','b','c','d']) #创建series（包含index的array）
obj['a'] #利用指数索引
obj[1]  #t同上作用 指数位数索引
obj[2:4] #2-3行
obj['b','c']
obj[['b','a','c']]
obj['b']=2 #赋值 广播
obj.index=['Bob','Steve','Jeff','Ryan'] #重新设置索引
obj[obj>2]#筛选
obj*2           #运算
np.exp(obj)     #运算
'e' in obj    #判断返回bool值
obj.isnull()    #判断是否为空值
obj.notnull()
obj.name='population'    #命名
obj.index.name='state'    #指数命名
obj.drop(['a','c'])   #按照index丢弃
obj2=obj.reindex(['a','b','c','d','e'],fill_value=0)#重排列索引 并赋空值为0
obj3=pd.Series(['blue','purple','yellow'],index=[0,2,4])
obj3.reindex(range(6),method='ffill')#向前填充
obj.unique() #唯一值
obj.index.is_unique #判断index是否是唯一
obj.value_counts() #不同值个数
mark = obj.isin(['b','c']) #返回一个布尔series 判断是否属于b c
obj[mark] #返回筛选出只含bc的series
obj.sort_index(axis=0,ascending=False) #按照index排序
obj.order()#默认排序 将NA值放在Series末尾


# %%
#dataframe 构建及修改 ========================================
data = {'state':['Ohio','Ohio','Ohio','Nevada','Nevada'],'year':[2000,2001,2002,2001,2002],'pop':[1.5,1.7,3.6,2.4,2.9]}
frame= pd.DataFrame(data)#由字典创建dataframe
pd.DataFrame(data,columns=['year','state','pop'],index=['a','b','c','d','e'])#重排序列及指数修改
frame['debt']=np.arange(5.) #新建列并赋值
frame['debt']=Series([-1.2,-1.5,-1.7],index=['a','c','e']) #新建列用series赋值
frame['eastern']=frame.state=='Ohio' #利用布尔值创建新列标记
frame.index.name='LETTER';frame.columns.name='INFOR' #给指数和列赋名字 属性
frame = pd.DataFrame(np.arange(9).reshape((3,3)),index=['a','b','c'],columns=['Ohio','Texas','California']) #设立df
frame2 = frame.reindex(index=['a','b','c','b'],colums=['Texas','Utah','California'],method='ffill') #重新索引行列 向前填充
frame2 = frame.ix[['a','b','c','b'],['Texas','Utah','California']] #作用同上无填充
frame2.drop(['Utah','California']) #丢弃列

# %%
#dataframe选取索引切片 ========================================
frame.values #返回ndarray类型的全部数据
frame['state'] #选取列 返回series
frame.state #选取列 返回series
frame.ix['a'] #按照指数选择行       
frame['Texas'] #列索引
frame[['Texas','California']] #多列索引
frame[:2] #0-1行
frame[(frame['Texas']>5)] #条件删选
frame<5 #返回布尔值df
# group by索引-------------------------------------
groups = df.groupby('Quarter')
groups = df.groupby(['Quarter','odd_even'])
groups.mean()
groups.max()
groups.size()
# ix索引-------------------------------------
data = pd.DataFrame(np.arange(16).reshape((4,4)),index=['Ohio','Colorado','Utah','New Work'],columns=['one','two','three','four'])
data.ix['Colorado',['two','three']] # 列，行
data.ix[['Colorado','Utah'],[3,0,1]] # 行， 列
data.ix[data.three>5,:3] #条件+列
# 层次化索引-------------------------------------
data = pd.Series(np.random.randn(10),index=[['a','a','a','b','b','b','c','c','d','d'],[1,2,3,1,2,3,1,2,2,3]])
data['b','c'] #选择索引为bc的行
data.ix[['b','c']]#同上
data[:,2] #内层索引
data.unstack() #将层次化索引重新安排到一个DataFrame中
data.unstack().stack() #将DataFrame转换为层次化索引
frame = pd.DataFrame(np.arange(12).reshape((4,3)),index = [['a','a','b','b'],[1,2,1,2]],columns=[['Ohio','Ohio','Colorado'],['Green','Red','Green']]) #多层次化索引df
frame.index.names = ['key1','key2'] #index名称
frame.columns.names = ['state','color'] #列名称
frame.swaplevel('key1','key2') #交换两个索引的级别
frame.swaplevel(0,1).sortlevel(0)
frame.sum(level='key2') #分级求和
frame.sum(level='color',axis=1)#分级求和
df = pd.DataFrame({'a':range(7),'b':range(7,0,-1),'c':['one','one','one','two','two','two','two'],'d':[0,1,2,0,1,2,3]})
df1 = df.set_index(['c','d'])
df.set_index(['c','d'],drop=False) #保留下设为index的列


#%%
#dataframe运算 ===========================================
df1.add(df2.fill_value = 0) #plus 用0填充空值
#sub div mul
f = lambda x: x.max()-x.min()
data.apply(f,axis=1)
def f(x): return Series([x.min(),x.max()],index=['min','max'])
data.apply(f)
format = lambda x: '%.2f'%x
data.applymap(format) # dataframe/seri.apply(lambda x:...)
data.sum(axis=1) #每行求和       
data.mean(axis=1,skipna=False)        
data.idxmax() #达到最小值或最大值的索引值
data.argmax() #达到最小值或最大值的索引位置
data.cumsum() #累加
 #quantile sum mean median mad var std skew偏度 kurt丰度 cummin cummax cumprod累计积 diff一阶差分       
returns = data.pct_change()
returns.corr()
returns.cov()    
returns.MSFT.corr(returns.IBM) #返回两列相关性
returns.corrwith(returns.IBM)  #一个sery和整个df的列的相关性
#df[u'移动均值']=pd.rolling_mean(df['Close'],window=42)   #计算移动指标
#df[u'移动std']=pd.rolling_std(df['Close'],window=42)   #计算移动指标
df[u'移动均值']=df.rolling(42).mean()
df[u'移动std']=df.rolling(42).std()


#%%
#dataframe处理缺省值 ===========================================        
data = pd.DataFrame(np.random.randn(7,3))
data.ix[:4,1] = np.nan # depreciated
data.ix[:2,2] = np.nan # depreciated
#.ix is deprecated. Please use
#.loc for label based indexing or
#.iloc for positional indexing
data.dropnna(axis=0) #除去含na的行 默认axis=0
data.dropna(thresh = 3) #同上
data.dropna(how='all') #除去全为na的行
data[data.notnull()] #布尔索引除去na
data.fillna(0,inplace=True)#填充缺失值
data.fillna({1:0.5,3:-1}) #对不同的列赋不同的值
data.fillna(method='ffill') #前值填充
data.fillna(method='ffill',limit=2) #前值填充 仅限2个NA       
data.replace([-999,-100],[np.nan,0]) #替换数据
data.replace({-999:np.nan,-100:0}) #功能同上

#%%
#dataframe数据规整化 ===========================================        
#合并数据集-------------------------------------
df = pd.merge(df1,df2,how="outer",on="localtime",suffixes=("_"+instru1,"_"+instru2)) #@axis=1(按照列合并)
#如果未指定on则merge就会将重叠列的列名当作键 how=='left','right','inner'
df = pd.merge(df1,df2,how="outer",left_on='lkey',right_on='rkey') #欲合并的列名不同   #@axis=1
df = pd.merge(df1,df2,on=['key1','key2'],how='outer') #按照两列合并                   #@axis=1
df = pd.merge(df1,df2,left_on='lkey',right_index=True) #左df的lkey和右df的index合并   #@axis=1
df1.join([df2,df3],how='outer',on='key') #在key列合并                                #@axis=1
np.concatenate([df1,df2],axis=1)                                                    #@axis=1
pd.concat([df1,df2,df3],axis=0)#默认axis=0                                          #@axis=0(按照行合并)
pd.concat([df1,df2,df3],axis=1,join='inner')                                        #@axis=1
pd.concat([df1,df2],axis=1,join_axes=['a','c','b','e'])# 指定在其他轴上使用的索引
pd.concat([df1,df2,df3],axis=1,keys=['one','two','three'])# 按照列合并
pd.concat([df1,df2],keys=['level1','level2'],names=['upper','lower']) #层次化索引的合并
pd.concat([df1,df2],ignore_index=True) #忽略指数
df1.combine_first(df2) #df1中的缺失值用df2对应的值补丁

#%%
#dataframe文件输入与输出 ===========================================
df0=pd.read_excel('E:\test.xlsx')
df01=df0.parse('Sheet1')#读取excel文件并选取相关表
df1=pd.read_csv('E:\test.csv',names=['a','b','c','d'],index_col='a')
df2=pd.read_csv('https://query1.finance.yahoo.com/v7/finance/download/ULAS.IS?period1=1491354787&period2=1493946787&interval=1d&events=history&crumb=UW5o55Kf2EK')#从网页批量下载csv文件
#参数: header 列名的行号 默认为0 没有为None skiprows 需要忽略的函数,从文件开始处算起 na_values一组用以替换NA的值
#parse_dates 尝试将数据解析为日期，也可指定多列 date_parser用于解析日期的函数 nrows 需要读取的行数 skip_fopter需要忽略的行数从文件末尾处算起
df1.to_csv(sys.stdout,na_rep='NULL') #sys.stout 仅仅print出来 代替缺省值
df2.to_csv(index=False,header=False) #不写入index和列名
sery.from_csv('E:/test.csv',parse_dates = True) #series快速导出并解析日期  
#---json文件-------------------------------------
result = json.loads(obj) #字典格式
asjson = json.dumps(result) #将python对象转换成JSON格式 
      
# %%
#Time Series ===========================================
#read_csv里存在时间日期解析
dates = pd.date_range('2015-1-1',periods=9,freq='M') #创建一列字符串格式的时间序列
#start=,end=,freq频率字符串=U微妙/L毫秒/S/T分钟/H小时/B交易日/D日历日/W/M/Q/A/BM月末交易日/MS月初/BMS月初交易日/
df.index = pd.to_datetime(df.index) # transfer index into datetime