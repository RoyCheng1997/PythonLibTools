
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 15:07:50 2017
This may not be useful since tushare has changed its version.
See: http://tushare.org/, For more information
@author: RoyCheng
"""
import tushare as ts   
import pandas as pd
import numpy as np             
target_stock_code = raw_input("Please input your stock code:")
target_stock_report_year = input("Please input the report time(year):")
target_stock_report_quarter = input("Please input the report time(quarter):")
transaction_day = raw_input("Please input the analysis day(YYYY-MM-DD):")
#location_save_all_data = raw_input("Please input location for saving:") #for writting




#==============================================================================基本面==============================================================================
total_basic_data = ts.get_stock_basics()
'''沪深上市公司基本情况'''
total_financial_report = ts.get_report_data(target_stock_report_year,target_stock_report_quarter)
'''按年度、季度获取业绩报表数据'''
total_profit_data = ts.get_profit_data(target_stock_report_year,target_stock_report_quarter)
'''盈利能力'''
total_operation_data = ts.get_operation_data(target_stock_report_year,target_stock_report_quarter)
'''营运能力'''
total_growth_data = ts.get_growth_data(target_stock_report_year,target_stock_report_quarter)
'''成长能力'''
total_leverage_data = ts.get_debtpaying_data(target_stock_report_year,target_stock_report_quarter)
'''偿债能力'''
total_cashflow_data = ts.get_cashflow_data(target_stock_report_year,target_stock_report_quarter)
'''现金流量'''
#==============================================================================历史数据==============================================================================
History_bar_df = ts.get_hist_data(target_stock_code,start='2015-01-05',end='2015-01-09') #前复权
'''获取近三年历史数据'''
# ts.get_hist_data('600848', ktype='W') #获取周k线数据
# ts.get_hist_data('600848', ktype='M') #获取月k线数据
# ts.get_hist_data('600848', ktype='5') #获取5分钟k线数据
# ts.get_hist_data('600848', ktype='15') #获取15分钟k线数据
# ts.get_hist_data('600848', ktype='30') #获取30分钟k线数据
# ts.get_hist_data('600848', ktype='60') #获取60分钟k线数据
# ts.get_hist_data('sh'）#获取上证指数k线数据，其它参数与个股一致，下同
# ts.get_hist_data('sz'）#获取深圳成指k线数据
# ts.get_hist_data('hs300'）#获取沪深300指数k线数据
# ts.get_hist_data('sz50'）#获取上证50指数k线数据
# ts.get_hist_data('zxb'）#获取中小板指数k线数据
# ts.get_hist_data('cyb'）#获取创业板指数k线数据
History_bar_df1 = ts.get_h_data(target_stock_code, start='2015-01-01', end='2015-03-16') #两个日期之间的前复权数据
'''获取全部历史数据'''
# ts.get_h_data('002337') #前复权
# ts.get_h_data('002337', autype='hfq') #后复权
# ts.get_h_data('002337', autype=None) #不复权
# ts.get_h_data('399106', index=True) #深圳综合指数
History_tick_df = ts.get_tick_data(target_stock_code,date='2014-01-09')
'''历史分笔数据明细(无bidask)'''
#==============================================================================即时数据==============================================================================
realTimeAll = ts.get_today_all()
'''当前交易所有股票的行情数据'''
realTimeIndex = ts.get_index()
'''当前大盘的行情数据'''
realtime_tran_data = ts.get_realtime_quotes(target_stock_code)
'''实时分笔数据'''
#ts.get_realtime_quotes(['600848','000980','000981']) #symbols from a list
#ts.get_realtime_quotes('sh')#上证指数
#ts.get_realtime_quotes(['sh','sz','hs300','sz50','zxb','cyb'])#上证指数 深圳成指 沪深300指数 上证50 中小板 创业板
#ts.get_realtime_quotes(['sh','600848'])#或者混搭
realday_deal_data = ts.get_today_ticks(target_stock_code)
'''当日历史分笔'''
big_deal_data = ts.get_sina_dd(target_stock_code, date=transaction_day)
'''获取大单交易数据，默认为大于等于400手'''
#df = ts.get_sina_dd('600848', date='2015-12-24', vol=500)  #指定大于等于500手的数据

M_recent_k_data = ts.get_k_data(target_stock_code, ktype='M')
'''最近月k线'''
W_recent_k_data = ts.get_k_data(target_stock_code, ktype='W')
'''最近周k线'''
D_recent_k_data = ts.get_k_data(target_stock_code, ktype='D')
'''最近日k线'''
recent_k60_data = ts.get_k_data(target_stock_code, ktype='60')
'''最近小时线'''
recent_k30_data = ts.get_k_data(target_stock_code, ktype='30')
'''最近30min线'''
recent_k15_data = ts.get_k_data(target_stock_code, ktype='15')
'''最近15min线'''
recent_k5_data = ts.get_k_data(target_stock_code, ktype='5')
'''最近5min线'''
#==============================================================================指数成分==============================================================================
'''获取指数成分'''
df_hs300 = ts.get_hs300s() #获取沪深300当前成份股及所占权重
df_sz50 = ts.get_sz50s() #获取上证50成份股
df_zz500 = ts.get_zz500s() #获取中证500成份股
#==============================================================================投资参考==============================================================================
df_dividend = ts.profit_data(top=60)
df_dividend.sort('shares',ascending=False)
'''分配预案'''
df_forecast = ts.forecast_data(2017,4)
'''业绩预告'''
df_restrict = ts.xsg_data()
'''限售股解禁'''
df_fund = ts.fund_holdings(2017, 2)
'''基金持股'''
df_sh = ts.sh_margin_details(start='2015-01-01', end='2015-04-19', symbol='601989')
'''上交所融资融券明细'''
df_sz = ts.sz_margins(start='2015-01-01', end='2015-04-19')
df_sz_detail = ts.sz_margin_details('2015-04-20')
'''深交所融资融券总额'''
#==============================================================================新闻==============================================================================
ts.get_latest_news() #默认获取最近80条新闻数据，只提供新闻类型、链接和标题
ts.get_latest_news(top=5,show_content=True) #显示最新5条新闻，并打印出新闻内容
'''即时新闻'''
ts.get_notices('600028')
'''个股新闻'''
ts.guba_sina(True)
'''新浪股吧'''
#==============================================================================海外数据源==============================================================================
import pandas.io.data as web
DAX = web.DataReader(name='^GDAXI',data_source='yahoo',start='2000-1-1')#读取德国DAX指数
#data_source='yahoo'/'google'/'fred'圣路易斯储蓄银行/'famafrench'/'pandas.io.web'世界银行  

#write_code
#total_basic_data.to_excel(location_save_all_data+"/Basic_report_for_all_stocks.xlsx")
#total_financial_report.to_excel(location_save_all_data+"/all_general_financial_report.xlsx")
#total_profit_data.to_excel(location_save_all_data+"/all_profit_data.xlsx")
#total_operation_data.to_excel(location_save_all_data+"/all_operation_data.xlsx")
#total_growth_data.to_excel(location_save_all_data+"/all_growth_data.xlsx")
#total_leverage_data.to_excel(location_save_all_data+"/all_leverage_data.xlsx")
#total_cashflow_data.to_excel(location_save_all_data+"/all_cashflow_data.xlsx")

#realtime_tran_data.to_excel(location_save_all_data+"/realtime_tran_data.xlsx")
#realday_deal_data.to_excel(location_save_all_data+"realday_deal_data.xlsx")
#big_deal_data.to_excel(location_save_all_data+"big_deal_data.xlsx")
#all_day_transaction_data.to_excel(location_save_all_data+"all_day_transaction_data.xlsx")
#M_recent_k_data.to_excel(location_save_all_data+"Month_recent_k_data.xlsx")
#W_recent_k_data.to_excel(location_save_all_data+"Week_recent_k_data.xlsx")
#recent_k60_data.to_excel(location_save_all_data+"Hour_recent_k_data.xlsx")
#recent_k30_data.to_excel(location_save_all_data+"Min30_recent_k_data.xlsx")
#recent_k15_data.to_excel(location_save_all_data+"Min15_recent_k_data.xlsx")
#recent_k5_data.to_excel(location_save_all_data+"Min5_recent_k_data.xlsx")



#realtime_tran_data = ts.get_realtime_quotes(rcode)
#0：name，股票名字
#1：open，今日开盘价
#2：pre_close，昨日收盘价
#3：price，当前价格
#4：high，今日最高价
#5：low，今日最低价
#6：bid，竞买价，即“买一”报价
#7：ask，竞卖价，即“卖一”报价
#8：volume，成交量 maybe you need do volume/100
#9：amount，成交金额（元 CNY）
#10：b1_v，委买一（笔数 bid volume）
#11：b1_p，委买一（价格 bid price）
#12：b2_v，“买二”
#13：b2_p，“买二”
#14：b3_v，“买三”
#15：b3_p，“买三”
#16：b4_v，“买四”
#17：b4_p，“买四”
#18：b5_v，“买五”
#19：b5_p，“买五”
#20：a1_v，委卖一（笔数 ask volume）
#21：a1_p，委卖一（价格 ask price）
#...
#30：date，日期；
#31：time，时间；

#realday_deal_data = ts.get_today_ticks(rcode)
#time：时间
#price：当前价格
#pchange:涨跌幅
#change：价格变动
#volume：成交手
#amount：成交金额(元)
#type：买卖类型【买盘、卖盘、中性盘】

#big_deal_data = ts.get_sina_dd(rcode, date=rday) #默认400手
#code：代码
#name：名称
#time：时间
#price：当前价格
#volume：成交手
#preprice ：上一笔价格
#type：买卖类型【买盘、卖盘、中性盘】

#all_day_transaction_data = ts.get_hist_data(rcode) #前复权
#date：日期
#open：开盘价
#high：最高价
#close：收盘价
#low：最低价
#volume：成交量
#price_change：价格变动
#p_change：涨跌幅
#ma5：5日均价
#ma10：10日均价
#ma20:20日均价
#v_ma5:5日均量
#v_ma10:10日均量
#v_ma20:20日均量
#turnover:换手率[注：指数无此项]







    






