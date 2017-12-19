# -*- coding: utf-8 -*-
"""
Created on Jul 22 15:28:14 2017

@author: Rebecca Cui
"""
from scipy import optimize
from scipy import stats
from scipy.stats import norm
import numpy as np

import matplotlib.pyplot as plt
from backtest import *
import os
import math
import pandas as pd


def drawCoverage(coverage, date1, win, save_path):##
    fig, ax = plt.subplots()
    ax = plt.plot(date1,coverage)
    coverage2=pd.Series(coverage).rolling(window=win,center=False).mean()
    fig=plt.plot(date1, coverage2,'r')
    plt.ylabel('# of stock')
    plt.title('Coverage')
    plt.ylim(0,max(coverage)+500)
    plt.savefig(save_path + 'Coverage')
    plt.show()  

def drawTurnover(Turnover, trd_freq, win, save_path):  ###
    """Calculates serial correlation and factor turnover from data of two adjacent months.
    """    
    fig, ax = plt.subplots()
    draw_turnover = Turnover[::trd_freq]
    date1 = pd.to_datetime(draw_turnover.index, format = "%Y%m%d")
    ax = plt.plot(date1,draw_turnover.ix[:,-1])
    turnover2=draw_turnover.ix[:,-1].rolling(window=52,center=False).mean()
    fig=plt.plot(date1,turnover2,'r')
    plt.ylim(0,4.0)
    
    plt.ylabel('Turnover')
    plt.title('Turnover')
    plt.savefig(save_path +'Turnover')
    plt.show()

   
def drawQuantileRtn(Portfolio_return,save_path):#drawQuantileRtn
    #Calculates and saves the annualized volatility of the portfolio.
    data = Portfolio_return.ix[:,:-1]
    data.index = pd.to_datetime(Portfolio_return.index,format = '%Y%m%d')

    #Calculates the annualized return of the portfolio, and saves it to the field.
    for i in range(len(Portfolio_return.columns)-1):
        net_rtn= list(Portfolio_return.ix[:,i])
        rtn = [1]
        for j in range(len(net_rtn)):
            rtn.append(rtn[-1]*(net_rtn[j]+1))
        data.ix[:,i]=rtn[1:]
        
    data.plot()
    plt.legend(loc=2,ncol=2,fancybox=True,shadow=True)
    #plt.ylim(0,max(data.ix[:,:-2])+1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize='small')    

    plt.ylabel('return')
    plt.title('Quantile Return')    
    plt.savefig(save_path +'Quantile Return')
    plt.show()
    data.to_csv(save_path + 'quantile_return.csv')
    return data

def drawExcessQuantileRtn(Portfolio_return,mkt_index,date1,save_path):#draw_quantile_rtn
    #Calculates and saves the annualized volatility of the portfolio.
    new_data = Portfolio_return.ix[:,:-1]
    for i in range(len(Portfolio_return.columns)-1):
        net_rtn = Portfolio_return.ix[:,i]-mkt_index.ix[:,"index"]
        rtn = [1]
        for day in net_rtn.index:
            rtn.append(rtn[-1]*(net_rtn[day]+1))
        rtn = pd.DataFrame(rtn[1:], index = net_rtn.index)
        new_data.ix[:,i] = rtn
    new_data.columns = Portfolio_return.ix[:,:-1].columns
    new_data.index = date1                                     

    new_data.plot()
    plt.legend(loc=2,ncol=2,fancybox=True,shadow=True)
    #plt.ylim(0,max(new_data.ix[:,-2])+1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize='small')    
    plt.ylabel('return')
    plt.title('Excess Quantile Return')    
    plt.savefig(save_path +'Excess Quantile Return')
    plt.show()
    new_data.to_csv(save_path + 'excess_quantile_return.csv')
    return new_data



def drawWealthCurve(Portfolio_return,save_path, IR_all,year,Anual_vol):
    lsnet = Portfolio_return.ix[:,-1][::5]
    lsdata = (lsnet+1).cumprod()
    
    #IR, MEAN, STDY
    
    plt.ylabel('Wealth Curve')
    plt.title('Wealth Curve')
    plt.plot(date1[::5],lsdata.values)
    length = max(lsdata)-min(lsdata)
    
    text1 = "IR: "+ str(round(IR_all[-1],2)) 
    text2 = "rtn: " + str(round((100*(lsdata.iloc[-1])**(48/len(lsdata))-100),2)) +" %"
    text3 = "std: " + str(round((np.std(100*lsnet)*math.sqrt(48.0)),2)) +" %"
                
    
    plt.text(datetime.datetime(year, 4, 4, 0, 0), max(lsdata), text1)  
    plt.text(datetime.datetime(year, 4, 4, 0, 0), max(lsdata)-0.08*length, text2)  
    plt.text(datetime.datetime(year, 4, 4, 0, 0), max(lsdata)-0.16*length, text3)
    
    
    plt.savefig(save_path +'Wealth Curve')   
    plt.show()
    return lsdata


def calAnualReturn(data,lsdata,save_path,labels):
    rtn_all = []
    #Calculates the annualized return of the portfolio, and saves it to the field.
    for i in range(len(data.columns)):
        rtn = data.ix[:,i]
        rtn_all.append(100*(rtn[-1])**(240.0/len(rtn))-100)
        #print rtn_all[-1]
    ls = lsdata.dropna()
    ls = (ls+1).cumprod()
    rtn=np.array(ls)
    rtn_all.append(100*(rtn[-1])**(48.0/len(rtn))-100)
    
    plt.ylim(min(rtn_all)-10, max(rtn_all)+10)
    plt.xticks(range(len(labels)),labels,rotation=90)
    plt.ylabel('Annualized Return(%)')
    plt.title('Annualized Return')
    plt.bar(range(len(labels)), rtn_all,align="center")
    plt.savefig(save_path +'Annualized Return')
    plt.show()
    return rtn_all


def calAnualVol(Portfolio_return,save_path,labels):
    #Calculates and saves the annualized volatility of the portfolio.
    std_all = []
    #Calculates the annualized return of the portfolio, and saves it to the field.
    for i in range(len(Portfolio_return.columns)-1):
        rtn = Portfolio_return.ix[:,i]*100
        std_all.append((np.std(rtn))*math.sqrt(240.0))
    rtn=Portfolio_return.ix[:,-1][::5]*100
    std_all.append((np.std(rtn))*math.sqrt(48.0))
    plt.bar(range(len(Portfolio_return.columns)), std_all, align="center")
    plt.xticks(range(len(labels)),labels,rotation=90)
    plt.ylabel('Annualized Volatility(%)')
    plt.title('Annualized Volatility(%)')
    plt.savefig(save_path +'Annualized Volatility')
    plt.show()
    return std_all

   
def drawSampleRtn(Portfolio_return,date1,save_path,win, year):
    #Calculates and saves the annualized volatility of the portfolio.
    time_series1=Portfolio_return.ix[:,-1][::5]*100
    time_series2 =time_series1.rolling(window=52,center=False).mean()
    fig12,p12=plt.subplots()
    date1 = pd.to_datetime(time_series1.index,format = '%Y%m%d')
    p12 = plt.bar(date1,time_series1,edgecolor = 'lightskyblue')
    fig12 =plt.plot(date1,time_series2,'r')
    plt.ylim(min(time_series1)-2,max(time_series1)+4)
    
        
    avg = np.mean(time_series1)
    std = np.std(time_series1)
    min_net_rtn = min(time_series1)
    res = avg/std
    
    text1='avg: '+ str(round(avg,2)) +" %"
    text2 = 'std: '+str(round(std,2)) +" %"
    text3 = 'min: ' + str(round(min_net_rtn,2)) +" %"
    text4 = 'avg/std: ' +str(round(res,2)) +" %"
    
    length = max(time_series1)-min(time_series1)+6
    
    plt.text(datetime.datetime(year, 4, 4, 0, 0), max(time_series1)+4-length*0.1, text1)  
    plt.text(datetime.datetime(year, 4, 4, 0, 0), max(time_series1)+4-length*0.16, text2)  
    plt.text(datetime.datetime(year, 4, 4, 0, 0), max(time_series1)+4-length*0.22, text3)  
    plt.text(datetime.datetime(year, 4, 4, 0, 0), max(time_series1)+4-length*0.28, text4)  
 
    plt.ylabel('Time Series Spread(%)')
    plt.title('Time Seires Spread(%)')
    plt.savefig(save_path +'Time Seires Spread(%)')
    plt.show()
    
    
#serial correlation
def calSerial(factor_data, date, date1,save_path,trd_freq, win,year):  ###
    """Calculates serial correlation and factor turnover from data of two adjacent months.
    """
    serial_cor=[]
    date2 = date[::trd_freq]
    date3 = date1[::trd_freq]
    for i in range(1,len(date2)):
        lastweek = date2[i-1]
        thisweek = date2[i]
        lastWeekData =factor_data.ix[lastweek]
        thisWeekData =factor_data.ix[thisweek]
        stocks = list(set(lastWeekData.dropna().index) & set(thisWeekData.dropna().index))
        last = lastWeekData.ix[stocks]
        this = thisWeekData.ix[stocks]
        mu1, mu2, sigma1, sigma2 = np.nanmean(last), np.nanmean(this), np.nanstd(last), np.nanstd(this)
        if (sigma1 * sigma2==0):
            serial = 100
        else:
            serial = np.nanmean((last - mu1) * (this - mu2))/float(sigma1 * sigma2) * 100
        # print (serial)
        serial_cor.append(serial)
        
    serial_cor = pd.Series(serial_cor)
    serial_cor.index = date2[1:]
    
    fig12,p12=plt.subplots()
    serial_cor2=serial_cor.rolling(window=52,center=False).mean()
    p12 = plt.plot(date3[1:],serial_cor)
    fig12 =plt.plot(date3[1:],serial_cor2,'r')
    plt.ylim(min(serial_cor)-10,max(serial_cor)+10)
            
    avg = np.mean(serial_cor)
    std = np.std(serial_cor)
    min_serial= min(serial_cor)
    res = avg/std
    
    text1='avg: '+ str(round(avg,2)) +" %"
    text2 = 'std: '+str(round(std,2)) +" %"
    text3 = 'min: ' + str(round(min_serial,2)) +" %" 
    text4 = 'avg/std: ' +str(round(res,2)) +" %"
    
    length = max(serial_cor) - min(serial_cor) + 20
            
    plt.text(datetime.datetime(year, 4, 4, 0, 0), max(serial_cor)+10-length*0.1, text1)  
    plt.text(datetime.datetime(year, 4, 4, 0, 0), max(serial_cor)+10-length*0.16, text2)  
    plt.text(datetime.datetime(year, 4, 4, 0, 0), max(serial_cor)+10-length*0.22, text3)  
    plt.text(datetime.datetime(year, 4, 4, 0, 0), max(serial_cor)+10-length*0.28, text4)  

    plt.ylabel('Serial corr(%)')
    plt.title('Serial Correlation(%)')
    plt.savefig(save_path +'Serial Correlation')
    plt.show()
    return serial_cor

def calIR(Portfolio_return,mkt_index,save_path):
    IR_all = []   
    #Calculates the annualized return of the portfolio, and saves it to the field.
    for i in range(len(Portfolio_return.columns)-1):
        rtn = Portfolio_return.ix[:,i]
        IR_all.append(((np.nanmean(rtn-mkt_index.ix[:,"index"]))/np.nanstd(rtn-mkt_index.ix[:,"index"]))*math.sqrt(240))
    
    ls_rtn = Portfolio_return.ix[:,-1][::5]
    IR_all.append((np.nanmean(ls_rtn)/np.nanstd(ls_rtn))*math.sqrt(48.0))
    plt.bar(range(len(Portfolio_return.columns)), IR_all, align="center")
    plt.ylabel('IR')
    plt.title('IR')
    plt.savefig(save_path +'IR')
    plt.show()
    return IR_all
    
def calSortinoRatio(Portfolio_return,mkt_index,save_path):
    SR_all = []
    for i in range(len(Portfolio_return.columns)-1):
        rtn = Portfolio_return.ix[:,i]
        DR=[]
        for day in rtn.index:
            DR.append(min(0,(rtn-mkt_index.ix[:,"index"])[day]))
        SR_all.append(math.sqrt(240)*(np.mean(rtn-mkt_index.ix[:,"index"]))/np.std(DR)) 
    ls_rtn = Portfolio_return.ix[:,-1][::5]
    DR= ls_rtn.copy()
    for day in ls_rtn.index:
        DR[day] = (min(0,ls_rtn[day]))
    SR_all.append(math.sqrt(48)*(np.mean(ls_rtn)/np.std(DR)))
    plt.bar(range(len(Portfolio_return.columns)), SR_all, align="center")
    plt.ylabel('Sortino Ratio')
    plt.title('Sortino Ratio')
    plt.savefig(save_path +'Sortino Ratio')
    plt.show()
    return SR_all

def calSpearman(factor_data, fwrtn,date, date1,save_path,win,year):
    Spearman_cor=[]
    for i in range(len(date)):#len(date1)):
        thisweek = date[i]
        thisWeekData =factor_data.ix[thisweek]
        rtndata = fwrtn.ix[thisweek]
        
        stocks = list(set(rtndata.dropna().index) & set(thisWeekData.dropna().index))
        #print (stocks)
        facs = thisWeekData.ix[stocks]
        fmrts = rtndata.ix[stocks]
        
        cor = stats.spearmanr(facs, fmrts).correlation * 100
        #print (cor)
        Spearman_cor.append(cor)
     
    Spearman_cor = pd.Series(Spearman_cor)
    Spearman_cor.index = date
    avg_IC = np.mean(Spearman_cor)
    std_IC = np.std(Spearman_cor)
    min_IC = min(Spearman_cor)
    res = avg_IC/std_IC
    
    text1='avg: '+ str(round(avg_IC,2)) + " %"
    text2 = 'std: '+str(round(std_IC,2)) + " %"
    text3 = 'min: ' + str(round(min_IC,2)) + " %"
    text4 = 'avg/std: ' +str(round(res,2)) 
                
    fig12,p12=plt.subplots()
    
    Spearman_cor.to_csv(save_path+"Spearman IC.csv")
    Spearman_cor2=Spearman_cor.rolling(window= win,center=False).mean()
    p12 = plt.bar(date1,Spearman_cor,edgecolor = 'lightskyblue')
    fig12 =plt.plot(date1,Spearman_cor2,'r')
    plt.ylim(min(Spearman_cor)-10,max(Spearman_cor)+10)
    length = max(Spearman_cor)-min(Spearman_cor)+20

    plt.text(datetime.datetime(year, 4, 4, 0, 0), max(Spearman_cor)+10-length*0.1, text1)  
    plt.text(datetime.datetime(year, 4, 4, 0, 0), max(Spearman_cor)+10-length*0.16, text2)  
    plt.text(datetime.datetime(year, 4, 4, 0, 0), max(Spearman_cor)+10-length*0.22, text3)  
    plt.text(datetime.datetime(year, 4, 4, 0, 0), max(Spearman_cor)+10-length*0.28, text4)  
        
    
    #plt.text(3, 8,'a', bbox={'avg':avg_IC,'std':std_IC, 'min':min_IC,'avg/std':res})    
    plt.ylabel('Spearman Corr(%)')
    plt.title('Spearman IC(%)')
    plt.savefig(save_path +'Spearman IC')
    
    plt.show()
    return Spearman_cor
'''
def calAdjSpearman(factor_data, rtn_adj, save_path):
    Spearman_cor = []
    date2 = rtn_adj.index
    for i in range(0, len(date2)):
        thisweek = date2[i]
        thisWeekData = factor_data.ix[thisweek]
        rtndata = rtn_adj.ix[thisweek]
        stocks = list(set(map(int,rtndata.dropna().index[:-1])) & set(thisWeekData.dropna().index))
        facs = thisWeekData.ix[stocks]
        fmrts = rtndata.ix[stocks]
        cor = stats.spearmanr(facs, fmrts).correlation * 100
        Spearman_cor.append(cor)

    Spearman_cor = pd.Series(Spearman_cor)
    Spearman_cor.index = date2[:]

    fig12, p12 = plt.subplots()
    Spearman_cor2 = Spearman_cor.rolling(window=52, center=False).mean()
    p12 = plt.bar(date2,Spearman_cor, edgecolor='lightskyblue')
    fig12 = plt.plot(Spearman_cor2, 'r')

    plt.ylabel('Spearman Corr(%)')
    plt.title('Spearman IC(%)')
    plt.savefig(save_path + 'Sector IC')
    return Spearman_cor
'''       

def drawICDecay(factor_data, fwrtn,date, date1,save_path,trd_freq):
    all_spearman =  []
    for j in range(0,12):
        Spearman_cor=[]
        for i in range(0,len(date1)-j*trd_freq):
            thisweek = date[i]
            thisWeekData =factor_data.ix[thisweek]
            fwddate = date[i+j*trd_freq]            
            rtndata = fwrtn.ix[fwddate]
            stocks = list(set(rtndata.dropna().index) & set(thisWeekData.dropna().index))
            facs = thisWeekData.ix[stocks]
            fmrts = rtndata.ix[stocks]
            cor = stats.spearmanr(facs, fmrts).correlation
            Spearman_cor.append(cor)
        all_spearman.append(np.nanmean(Spearman_cor))
        
    plt.bar(range(1,13),np.array(all_spearman)*100)

    plt.ylabel('Spearman Corr %')
    plt.title('IC Decay')
    plt.savefig(save_path +'IC Decay')
    plt.show()



def drawICDist(Spearman_cor,date1,save_path):
    mean = np.nanmean(Spearman_cor/100.0)
    std = np.nanstd( Spearman_cor/100.0)
    a=list(pd.Series(Spearman_cor).dropna()/100.0)
    Y = norm(loc=mean,scale=std)
    t = np.arange(np.nanmin(a)-0.2,np.nanmax(a)+0.2,0.01)
    
    fig12,p12=plt.subplots()    
    p12 = plt.hist(a,bins=40,normed=1, facecolor='blue', alpha=0.5)  
    fig12 =plt.plot(t,Y.pdf(t),'r')

    plt.text(np.nanmin(a)-0.15, 1.0/(math.sqrt(math.pi*2)*std) +0.3 , '$\mu$ = '+str(round(mean,2)))  
    plt.text(np.nanmin(a)-0.15, 1.0/(math.sqrt(math.pi*2)*std) , '$\sigma$ = '+str(round(std,2))) 
    plt.text(np.nanmin(a)-0.15, 1.0/(math.sqrt(math.pi*2)*std) -0.3, '$pval$ = '+str(round(stats.normaltest(a)[1],3)))  
    plt.xlabel('Spearman Corr')
    plt.title('Spearman IC distribution ')
    plt.savefig(save_path +'Spearman IC distribution')
    plt.show()
