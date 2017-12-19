#!/usr/bin/env python
# coding=utf-8
# python 2.7.12
from backtestclass import backtest
import pandas as pd
import os
k = 0
Fname=[]
Fdata=[]
# 从secondfactor中读取所有的因子表
path = r'C:\DELL\quantitative research\Firtest\secondfactor\\'
for file in os.listdir(path):
    Fname.append(file)
    df = pd.read_csv(path + file, sep=',', index_col=0)
    Fdata.append(df)
F = pd.Series(index=Fname, data=Fdata)

# 构造转换成时间序列的函数
def timeconvert(x):
    x.index = x.index.astype('str')
    x.index = pd.to_datetime(pd.DatetimeIndex(x.index).date)
# 从数据的第二周开始，因为第一个交易周期的收益数据不能获取，所以所有的数据处理范围都是从第二周期开始到最后一周期的
def nor(x):
    return x.iloc[1:,:]
# 构造计算收益率的函数
def Return(x):
    x1=x.iloc[:-1,:]
    x2=x.iloc[1:,:]
    x1.index=x2.index
    return (x2/x1) -1

#读取原始数据表（这里只用算收益率，所以只读取权数表，收盘价和沪深指数表
DCloP = pd.read_csv("C:\DELL\quantitative research\Firtest\Origidata\ClosePrice_20070131_20130301.csv", sep=',', index_col=0)
HS =  pd.read_csv("C:\DELL\quantitative research\Firtest\Origidata\hs_zz_zz_20070131_20130301.csv", sep=',', index_col=0)
timeconvert(DCloP)
timeconvert(HS)

#将交易日改为以5，10，20，60天为周期
DCloP_5  = (DCloP.resample('W',label='left').last()).dropna(how='all')
DCloP_10  = (DCloP.resample('2W',label='left').last()).dropna(how='all')
DCloP_20 = (DCloP.resample('4W',label='left').last()).dropna(how='all')
DCloP_60  = (DCloP.resample('12W',label='left').last()).dropna(how='all')
HS5 = (HS.resample('W',label='left').last()).dropna(how='all')
HS10 = (HS.resample('2W',label='left').last()).dropna(how='all')
HS20 = (HS.resample('4W',label='left').last()).dropna(how='all')
HS60 = (HS.resample('12W',label='left').last()).dropna(how='all')

#算收益率
[HS_5,HS_10,HS_20,HS_60] = [Return(HS5),Return(HS10),Return(HS20),Return(HS60)]
[FR_5,FR_10,FR_20,FR_60] = [Return(DCloP_5),Return(DCloP_10),Return(DCloP_20),Return(DCloP_60)]

#构造循环遍历的参数
list = pd.Series(index=F.index)
cnt = 0
frs = [FR_10, FR_20, FR_5, FR_60]    #存所有交易周期return的list
hss = [HS_10, HS_20, HS_5, HS_60]    #存所有交易周期benchmark的list
num = [2, 4, 1, 12]                  #存周的倍数
list = {}
for label in F.index:
    i=0
    FACname = ''
    ARGname = 'P'
    while(label[i]!='_'):
        FACname = FACname + label[i]     #在表名中提取出因子的名称
        i = i+1
    i=i+1
    while(label[i]!='.'):
        ARGname = ARGname + label[i]     #在表名中提取数字（5，10，20，60）
        i = i+1
    newtest = backtest((F[label]).T,(frs[cnt]).T,hss[cnt],num[cnt])        #定义对象
    list[label]=newtest
    (list[label]).cal_all(FACname, ARGname)
    cnt = cnt + 1
    cnt = cnt % 4



