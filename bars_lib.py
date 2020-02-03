# -*- coding: utf-8 -*-
"""
These demos show the way to plot different kinds of bar. 
Resources are from Internet.
"""

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, WeekdayLocator,\
    DayLocator, MONDAY,date2num
#from matplotlib.finance import  candlestick_ohlc
from mpl_finance import candlestick_ohlc
# %%
#    Classes Form  
class BarSeries(object):
    '''
        Base class for resampling ticks dataframe into OHLC(V)
        using different schemes. This particular class implements
        standard time bars scheme.
        See: https://www.wiley.com/en-it/Advances+in+Financial+Machine+Learning-p-9781119482086
    '''
    
    def __init__(self, df, datetimecolumn = 'DateTime'):
        self.df = df
        self.datetimecolumn = datetimecolumn
    
    def process_column(self, column_name, frequency):
        return self.df[column_name].resample(frequency, label='right').ohlc()
    
    def process_volume(self, column_name, frequency):
        return self.df[column_name].resample(frequency, label='right').sum()
    
    def process_ticks(self, price_column = 'Price', volume_column = 'Size', frequency = '15Min'):
        price_df = self.process_column(price_column, frequency)
        volume_df = self.process_volume(volume_column, frequency)
        price_df['volume'] = volume_df
        return price_df
    

class TickBarSeries(BarSeries):
    '''
        Class for generating tick bars based on bid-ask-size dataframe
    '''
    def __init__(self, df, datetimecolumn = 'DateTime', volume_column = 'Size'):
        self.volume_column = volume_column
        super(TickBarSeries, self).__init__(df, datetimecolumn)
    
    def process_column(self, column_name, frequency):
        res = []
        for i in range(frequency, len(self.df), frequency):
            sample = self.df.iloc[i-frequency:i]
            v = sample[self.volume_column].values.sum()
            o = sample[column_name].values.tolist()[0]
            h = sample[column_name].values.max()
            l = sample[column_name].values.min()
            c = sample[column_name].values.tolist()[-1]
            d = sample.index.values[-1]
            
            res.append({
                self.datetimecolumn: d,
                'open': o,
                'high': h,
                'low': l,
                'close': c,
                'volume': v
            })

        res = pd.DataFrame(res).set_index(self.datetimecolumn)
        return res

    
    def process_ticks(self, price_column = 'Price', volume_column = 'Size', frequency = '15Min'):
        price_df = self.process_column(price_column, frequency)
        return price_df    
    

class VolumeBarSeries(BarSeries):
    '''
        Class for generating volume bars based on bid-ask-size dataframe
    '''
    def __init__(self, df, datetimecolumn = 'DateTime', volume_column = 'Size'):
        self.volume_column = volume_column
        super(VolumeBarSeries, self).__init__(df, datetimecolumn)
               
    def process_column(self, column_name, frequency):
        res = []
        buf = []
        start_index = 0.
        volume_buf = 0.
        for i in range(len(self.df[column_name])):

            pi = self.df[column_name].iloc[i]
            vi = self.df[self.volume_column].iloc[i]
            di = self.df.index.values[i]
            
            buf.append(pi)
            volume_buf += vi
            
            if volume_buf >= frequency:
                
                o = buf[0]
                h = np.max(buf)
                l = np.min(buf)
                c = buf[-1]
                
                res.append({
                    self.datetimecolumn: di,
                    'open': o,
                    'high': h,
                    'low': l,
                    'close': c,  
                    'volume': volume_buf
                })
                
                buf, volume_buf = [], 0.

        res = pd.DataFrame(res).set_index(self.datetimecolumn)
        return res
    
    def process_ticks(self, price_column = 'Price', volume_column = 'Size', frequency = '15Min'):
        price_df = self.process_column(price_column, frequency)
        return price_df    
    
    

class DollarBarSeries(BarSeries):
    '''
        Class for generating "dollar" bars based on bid-ask-size dataframe
    '''
    def __init__(self, df, datetimecolumn = 'DateTime', volume_column = 'Size'):
        self.volume_column = volume_column
        super(DollarBarSeries, self).__init__(df, datetimecolumn)
               
    def process_column(self, column_name, frequency):
        res = []
        buf, vbuf = [], []
        start_index = 0.
        dollar_buf = 0.
        for i in range(len(self.df[column_name])):

            pi = self.df[column_name].iloc[i]
            vi = self.df[self.volume_column].iloc[i]
            di = self.df.index.values[i]
            
            dvi = pi * vi
            buf.append(pi)
            vbuf.append(vi)
            dollar_buf += dvi
            
            if dollar_buf >= frequency:
                
                o = buf[0]
                h = np.max(buf)
                l = np.min(buf)
                c = buf[-1]
                v = np.sum(vbuf)
                
                res.append({
                    self.datetimecolumn: di,
                    'open': o,
                    'high': h,
                    'low': l,
                    'close': c,
                    'volume': v,
                    'dollar': dollar_buf
                })
                
                buf, vbuf, dollar_buf = [], [], 0.

        res = pd.DataFrame(res).set_index(self.datetimecolumn)
        return res 
    
    def process_ticks(self, price_column = 'Price', volume_column = 'Size', frequency = '15Min'):
        price_df = self.process_column(price_column, frequency)
        return price_df    
    
    
class ImbalanceTickBarSeries(BarSeries):
    '''
        Class for generating imbalance tick bars based on bid-ask-size dataframe
    '''
    def __init__(self, df, datetimecolumn = 'DateTime', volume_column = 'Size'):
        self.volume_column = volume_column
        super(ImbalanceTickBarSeries, self).__init__(df, datetimecolumn)
        
    def get_bt(self, data):
        s = np.sign(np.diff(data))
        for i in range(1, len(s)):
            if s[i] == 0:
                s[i] = s[i-1]
        return s

    def get_theta_t(self, bt):
        return np.sum(bt)

    def ewma(self, data, window):

        alpha = 2 /(window + 1.0)
        alpha_rev = 1-alpha

        scale = 1/alpha_rev
        n = data.shape[0]

        r = np.arange(n)
        scale_arr = scale**r
        offset = data[0]*alpha_rev**(r+1)
        pw0 = alpha*alpha_rev**(n-1)

        mult = data*pw0*scale_arr
        cumsums = mult.cumsum()
        out = offset + cumsums*scale_arr[::-1]
        return out
               
    def process_column(self, column_name, initital_T = 100, min_bar = 10, max_bar = 1000):
        init_bar = self.df[:initital_T][column_name].values.tolist()

        ts = [initital_T]
        bts = [bti for bti in self.get_bt(init_bar)]  
        res = []

        buf_bar, vbuf, T = [], [], 0.
        for i in range(initital_T, len(self.df)):

 
            di = self.df.index.values[i]

            buf_bar.append(self.df[column_name].iloc[i])
            bt = self.get_bt(buf_bar)
            theta_t = self.get_theta_t(bt)

            try:
                e_t = self.ewma(np.array(ts), initital_T / 10)[-1]
                e_bt = self.ewma(np.array(bts), initital_T)[-1]
            except:
                e_t = np.mean(ts)
                e_bt = np.mean(bts)
            finally:                   
                if np.isnan(e_bt):
                    e_bt = np.mean(bts[int(len(bts) * 0.9):])
                if np.isnan(e_t):
                    e_t = np.mean(ts[int(len(ts) * 0.9):])

                
            condition = np.abs(theta_t) >= e_t * np.abs(e_bt)

            
            if (condition or len(buf_bar) > max_bar) and len(buf_bar) >= min_bar:

                o = buf_bar[0]
                h = np.max(buf_bar)
                l = np.min(buf_bar)
                c = buf_bar[-1]
                v = np.sum(vbuf)
                
                res.append({
                    self.datetimecolumn: di,
                    'open': o,
                    'high': h,
                    'low': l,
                    'close': c,
                    'volume': v
                })
                
                ts.append(T)
                for b in bt:
                    bts.append(b) 
                    
                buf_bar = []
                vbuf = []
                T = 0.           
            else:
                vbuf.append(self.df[self.volume_column].iloc[i])
                T += 1

        res = pd.DataFrame(res).set_index(self.datetimecolumn)
        return res 
    
    def process_ticks(self, price_column = 'Price', volume_column = 'Size', init = 100, min_bar = 10, max_bar = 1000):
        price_df = self.process_column(price_column, init, min_bar, max_bar)
        return price_df  
    
    
# %%
#    Functions Form  
def candlePlot(seriesData,title="a"):
	#设定日期格式
    Date=[date2num(date) for date in seriesData.index]
    seriesData.loc[:,'Date']=Date
    listData=[]
    for i in range(len(seriesData)):
        a=[seriesData.Date[i],\
        seriesData.Open[i],seriesData.High[i],\
        seriesData.Low[i],seriesData.Close[i]]
        listData.append(a)

	#设定绘图相关参数
    ax = plt.subplot()
    mondays = WeekdayLocator(MONDAY)
    #日期格式为‘15-Mar-09’形式
    weekFormatter = DateFormatter('%y %b %d')
    ax.xaxis.set_major_locator(mondays)
    ax.xaxis.set_minor_locator(DayLocator())
    ax.xaxis.set_major_formatter(weekFormatter)

	#调用candlestick_ohlc函数
    candlestick_ohlc(ax,listData, width=0.7,\
                     colorup='r',colordown='g')
    ax.set_title(title) #设定标题
    #设定x轴日期显示角度
    plt.setp(plt.gca().get_xticklabels(), \
    rotation=50,horizontalalignment='center')
    return(plt.show())

#蜡烛图与线图
def candleLinePlots(candleData, candleTitle='a', **kwargs):
    Date = [date2num(date) for date in candleData.index]
    candleData.loc[:,'Date'] = Date
    listData = []
    
    for i in range(len(candleData)):
        a = [candleData.Date[i],\
            candleData.Open[i],candleData.High[i],\
            candleData.Low[i],candleData.Close[i]]
        listData.append(a)
    # 如 果 不 定 长 参 数 无 取 值 ， 只 画 蜡 烛 图
    ax = plt.subplot()
    
    # 如 果 不 定 长 参 数 有 值 ， 则 分 成 两 个 子 图
    flag=0
    if kwargs:
        if kwargs['splitFigures']:
            ax = plt.subplot(211)
            ax2= plt.subplot(212)
            flag=1;
        # 如 果 无 参 数 splitFigures ， 则 只 画 一 个 图 形 框
        # 如 果 有 参 数 splitFigures ， 则 画 出 两 个 图 形 框
        for key in kwargs:
            if key=='title':
                ax2.set_title(kwargs[key])
            if key=='ylabel':
                ax2.set_ylabel(kwargs[key])
            if key=='grid':
                ax2.grid(kwargs[key])
            if key=='Data':
                plt.sca(ax)
                if flag:
                    plt.sca(ax2)
                    
                #一维数据
                if kwargs[key].ndim==1:
                    plt.plot(kwargs[key],\
                             color='k',\
                             label=kwargs[key].name)
                    plt.legend(loc='best')
                #二维数据有两个columns
                elif all([kwargs[key].ndim==2,\
                          len(kwargs[key].columns)==2]):
                    plt.plot(kwargs[key].iloc[:,0], color='k', 
                             label=kwargs[key].iloc[:,0].name)
                    plt.plot(kwargs[key].iloc[:,1],\
                             linestyle='dashed',\
                             label=kwargs[key].iloc[:,1].name)
                    plt.legend(loc='best')
    
    mondays = WeekdayLocator(MONDAY)
    weekFormatter = DateFormatter('%y %b %d')
    ax.xaxis.set_major_locator(mondays)
    ax.xaxis.set_minor_locator(DayLocator())
    ax.xaxis.set_major_formatter(weekFormatter)
    plt.sca(ax)
    
    candlestick_ohlc(ax,listData, width=0.7,\
                     colorup='r',colordown='g')
    ax.set_title(candleTitle)
    plt.setp(ax.get_xticklabels(),\
             rotation=20,\
             horizontalalignment='center')
    ax.autoscale_view()
    
    return(plt.show())

#蜡烛图与成交量柱状图
def candleVolume(seriesData,candletitle='a',bartitle='b',cols=['Open','High','Low','Close','Volume']):
    Date=[date2num(date) for date in seriesData.index]
    seriesData.index=list(range(len(Date)))
    seriesData['Date']=Date
    listData=zip(seriesData.Date,seriesData[cols[0]],seriesData[cols[1]],seriesData[cols[2]],
                 seriesData[cols[3]])
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    for ax in ax1,ax2:
        mondays = WeekdayLocator(MONDAY)
        weekFormatter = DateFormatter('%m/%d/%Y')
        ax.xaxis.set_major_locator(mondays)
        ax.xaxis.set_minor_locator(DayLocator())
        ax.xaxis.set_major_formatter(weekFormatter)
        ax.grid(True)
    ax1.set_ylim(seriesData[cols[2]].min()-2,seriesData[cols[1]].max()+2)
    ax1.set_ylabel('蜡烛图及收盘价线')
    candlestick_ohlc(ax1,listData, width=0.7,colorup='r',colordown='g')
    plt.setp(plt.gca().get_xticklabels(),\
            rotation=45,horizontalalignment='center')
    ax1.autoscale_view()
    ax1.set_title(candletitle)
    ax1.plot(seriesData.Date,seriesData[cols[3]],\
               color='black',label='收盘价')
    ax1.legend(loc='best')
    ax2.set_ylabel('成交量')
    ax2.set_ylim(0,seriesData[cols[4]].max()*3)
    ax2.bar(np.array(Date)[np.array(seriesData[cols[3]]>=seriesData[cols[0]])]
    ,height=seriesData.iloc[:,4][np.array(seriesData[cols[3]]>=seriesData[cols[0]])]
    ,color='r',align='center')
    ax2.bar(np.array(Date)[np.array(seriesData[cols[3]]<seriesData[cols[0]])]
    ,height=seriesData.iloc[:,4][np.array(seriesData[cols[3]]<seriesData[cols[0]])]
    ,color='g',align='center')
    ax2.set_title(bartitle)
    return(plt.show())

# =============================================================================
# ssec2012=pd.read_csv('dat\\ssec2012.csv')
# ssec2012.index=ssec2012.iloc[:,1]
# ssec2012.index=pd.to_datetime(ssec2012.index, format='%Y/%m/%d')
# ssec2012=ssec2012.iloc[:,2:]
# #Data1 = ssec2012['Close']
# candlePlot(ssec2012,title="Candle Plot")
# candleLinePlots(ssec2012,candleTitle="Candle Plot")
# =============================================================================
