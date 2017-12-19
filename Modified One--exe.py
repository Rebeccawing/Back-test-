#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Jul 25 11:17:21 2017

@author: Rebecca Cui
"""

import numpy as np
import pandas as pd
#from config import *


# data: rowname: date colname : stocks
class Backtest(object):
    def __init__(self, factor, FMRT, WEEKFMRT, path, factor_path, direction, trd_freq = 5, commision = 0.00025, stamp_tax = 0.001, impact_cost = 0.001):      
        self.data = pd.read_csv(factor_path + factor +'.csv', index_col = 0)
        self.trd_freq = trd_freq    
        self.fwrtn = pd.read_csv( path + FMRT +'.csv', index_col = 0)  
        self.weekfwrtn = pd.read_csv( path + WEEKFMRT +'.csv', index_col = 0) 
        self.data =  self.data.replace([np.inf,-np.inf], np.nan)

        self.stocks =  self.data.columns
        self.commision = commision
        self.stamp_tax = stamp_tax
        self.cost = impact_cost
             
        self.hs300_comp = pd.read_csv(path+'hs300_comp.csv', index_col = 0)
        self.zz500_comp = pd.read_csv(path+'zz500_comp.csv', index_col = 0)
        self.zz800_comp = pd.read_csv(path+'zz800_comp.csv', index_col = 0)

        if direction == "Descending":
            self.data = self.data*(-1)  
            
        #self.data = self.data.fillna(method = "ffill", limit=60)
        
    def dateIndex(self, start = '20160101', end = '20161231'):
        self.date=list(set(self.data.ix[start:end].index) & set(self.fwrtn.ix[start:end].index))                 #取交集
        self.date.sort()
        self.data = self.data.ix[self.date]
        self.fwrtn = self.fwrtn.ix[self.date]
        self.weekfwrtn = self.weekfwrtn.ix[start:end]
        self.weekfwrtn =  self.weekfwrtn[::self.trd_freq]
    
    #choose stock for one quantile, need to call n times for n quantile.
    def choose_stocks(self, day, choose_quantile, quantile, comp = 'ALL'):
        stockpool=list(set(self.data.ix[day].dropna().index)& set(self.fwrtn.ix[day].dropna().index))
        
        if (comp =='HS300'):
            stock_pool=self.hs300_comp.ix[day][self.hs300_comp.ix[day]==1].index
        elif(comp=='ZZ500'):
            stock_pool=self.zz500_comp.ix[day][self.zz500_comp.ix[day]==1].index
        elif(comp=='ZZ800'):
            stock_pool=self.zz500_comp.ix[day][self.zz800_comp.ix[day]==1].index
        elif(comp=='ALL'):
            stock_pool=stockpool
        #确定股票池    
        stockspool= list(set(stockpool) & set(stock_pool))
        stockspool.sort()     
        #对当天数据切片
        today_data=self.data.ix[day,stockspool]
        #正序，越小rank越小
        rank_data=np.ceil((today_data).rank()*1.0/(len(today_data))*quantile)
        #选取choose quantile的股票
        stocks = rank_data.ix[rank_data==choose_quantile].index
        return stocks
    
    #save_path: 存储路径  comp:股票池 fwrtn:rtn数据 freq:交易频率 longshort : 1 long 2 short 
    def Portfolio (self ,save_path, comp, fwrtn, freq, choose_quantile, quantile, longshort, cost1, cost2):
        #储存收益率
        self.net_rtn = []  
        #value是每天每只股票的价值
        value = pd.DataFrame()
        self.cover = []
        turnover=[2]
        if (longshort ==2):
            fwrtn = fwrtn*-1
        for i in range(len(fwrtn.index)):
            #print (i)
            day=fwrtn.index[i]
            if (i==0):
                #print 1
                #选取股票
                stocks = self.choose_stocks(day, choose_quantile, quantile, comp)
                chosenum = len(stocks)
                #初始化value
                perc=np.ones(chosenum)*1.0/chosenum
                perc=pd.Series(perc, index = stocks)
                #扣除交易费用，注意到利用(1.0-cost1)近似1/(1.0+cost1)
                rtn2 = np.nansum(perc*(1 + fwrtn.ix[day,stocks]))*float(1.0-cost1)-1
                #更新value
                perc = perc*(1+fwrtn.ix[day,stocks])*float(1.0-cost1)
                self.net_rtn.append(float(rtn2))
                value = pd.concat([value,perc],axis=1)
                self.cover.append(len(perc.dropna()))
                #print (self.cover[-1]) 

            #持仓日 
            elif (i%freq!=0):
                #print (2)
                rtn2 = np.nansum(perc*(1 + fwrtn.ix[day,stocks]))/np.nansum(perc)-1.0
                perc = perc*(1 + fwrtn.ix[day,stocks])
                self.net_rtn.append((rtn2))
                value = pd.concat([value,perc],axis=1)
                self.cover.append(len(perc.dropna()))
                turnover.append(0)
                #print (self.cover[-1])

            #调仓日：调仓前*（1-cost)*调仓后
            elif (i%freq==0):
                #print (3)
                old_perc=perc[:]
                #计算已有的value
                old_value = np.nansum(old_perc)

                #选取新股票，不考虑ST,NT,涨跌停
                new_stocks = self.choose_stocks(day, choose_quantile, quantile, comp)
                if len(stocks)== 0 and len(new_stocks)>0 :
                    old_value = 1.0

                #避免全天因子缺失
                if(len(new_stocks)==0):
                    new_stocks = stocks[:]
                if(len(new_stocks) > 1500):
                    new_stocks = stocks[:]
                    
                    
                    
                #高估v'                
                v =  old_value/len(new_stocks)
                #微调部分 hold_stocks 
                hold_stocks = list((set(stocks) & set(new_stocks)))
                hold_perc = old_perc.ix[hold_stocks] 
                #卖出部分
                sell_stocks = list(set(stocks)-set(hold_stocks))
                sell_value = np.nansum(old_perc.ix[sell_stocks])
                print (len(hold_stocks),len(sell_stocks))
                
                
                #可能被高估的value 千一数量级 迭代一次
                new_value = old_value - np.nansum(hold_perc.ix[old_perc >= v]-v)*(1-(1.0-cost1)*(1.0-cost2)) - sell_value*(1-(1.0-cost1)*(1.0-cost2))
                v1 = new_value/len(new_stocks)
                v=v1
                
                s = len(hold_perc.ix[hold_perc >= v])#卖一点
                t = len(hold_perc.ix[hold_perc < v])#买一点
                m = len(new_stocks) - len(hold_stocks)#建仓
                
                #在等权的情况下解除value总值
                new_value = (s+m+t)*(old_value - np.nansum(hold_perc.ix[old_perc >= v])*(cost2) + np.nansum(hold_perc.ix[old_perc < v])*(cost1) - sell_value*(cost2))/((1-cost2)*s + (1+cost1)*t + (1+cost1)*m)
                
                #更新价值
                #v = new_value/len(new_stocks)
                new_perc = np.ones(len(new_stocks))*new_value/len(new_stocks)
                
                #更新股票
                stocks = new_stocks[:]
                
                #print ("*", len(hold_perc.ix[hold_perc >= v]),len(hold_perc.ix[hold_perc < v]))
                #print ("**", len(hold_perc.ix[hold_perc >= new_value/len(new_stocks)])-len(hold_perc.ix[hold_perc >= v1]))
                
                perc=pd.Series(new_perc, index = new_stocks)
                
                rtn2 = np.nansum(perc*(1+fwrtn.ix[day,stocks]))/old_value -1
                perc = perc*(1+fwrtn.ix[day,stocks])
                self.net_rtn.append(float(rtn2))
                value = pd.concat([value,perc],axis=1)
                self.cover.append(len(perc.dropna())) #计算coverage，对于每个quantile
                print (self.cover[-1])
                
                #计算换手率，old_value是调仓前一天的总市值
                #turn = (np.nansum(hold_perc.ix[old_perc >= v]-v) + sell_value)*(1+(1-cost1)*(1-cost2))/old_value
                turn = (np.nansum(hold_perc.ix[old_perc >= v]-v) + sell_value)/old_value + (np.nansum(hold_perc.ix[old_perc >= v]-v) + sell_value)*(1-cost1)*(1-cost2)/old_value
                if len(stocks)== 0 :
                    turn = 0
                turnover.append(turn)
                #print (turn)
            #print (self.net_rtn[-1])

        value.columns = fwrtn.index
        #value.to_csv(save_path+'value'+str(quantile+1-choose_quantile)+'.csv')    
        
        net_rtn=pd.Series(self.net_rtn)
        net_rtn.index = fwrtn.index
        
        turnover = pd.Series(turnover)
        turnover.index = fwrtn.index       
        return net_rtn,turnover,value

    def MakePortfolio(self, save_path, comp="ALL", quantile = 5):
        Portfolio_return = pd.DataFrame()
        col_name = []
        coverage = np.array([0]*(len((self.date))))
        stocks_turnover = pd.DataFrame()
        
        #对每个quantile计算收益率曲线
        #使用daily数据，计算long only 的quantile曲线
        for j in range(quantile):
            col_name.append('Q'+ str(1+j))
            net_return,turnover,value = self.Portfolio(save_path, comp, self.fwrtn, self.trd_freq, (1+j), quantile, longshort = 1, cost1 = self.commision + self.cost, cost2 = self.commision + self.cost + self.stamp_tax )
            value.to_csv(save_path+'value'+str(quantile-j)+'.csv')    
            Portfolio_return = pd.concat([Portfolio_return,net_return],axis=1)
            stocks_turnover = pd.concat([stocks_turnover,turnover],axis=1)
            coverage = coverage + np.array(self.cover)
            
        #cost1是建仓费用（long的时候是买的手续费，short的时候是卖的手续费）, cost2平仓费用 
        #使用weekly的数据，避免持仓日计算再投资的错误
        long_weekly_rtn,long_weekly_turnover,long_value = self.Portfolio(save_path, comp, self.weekfwrtn, 1, quantile, quantile, longshort = 1, cost1 = self.commision + self.cost , cost2 = self.commision + self.cost + self.stamp_tax )
        short_weekly_rtn,short_weekly_turnover,short_value = self.Portfolio(save_path, comp, self.weekfwrtn, 1, 1, quantile, longshort = 2, cost1 = self.commision + self.cost + self.stamp_tax , cost2 = self.commision + self.cost)
        ls_return = long_weekly_rtn + short_weekly_rtn
        ls_turnover = long_weekly_turnover + short_weekly_turnover
        
        #拼接数据（注意最后一列有na)
        Portfolio_return = pd.concat([Portfolio_return,ls_return],axis=1)    
        stocks_turnover = pd.concat([stocks_turnover,ls_turnover],axis=1)
        
        #行列命名
        col_name.append('QL/S')
        Portfolio_return.columns = col_name       
        stocks_turnover.columns = col_name
        return Portfolio_return, coverage, stocks_turnover
            


