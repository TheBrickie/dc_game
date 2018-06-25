# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 20:38:11 2018
最初尝试：用线性模型预测，仅使用avg_online_minutes这一列数据作为特征向量输入
@author: 陈搏
"""

import pandas as pd
import numpy as np
import time
from sklearn import linear_model

DATAFOLDER=r'E:\works\python-works\dc_game\data'

def load_data():
    table_train=pd.read_csv(DATAFOLDER+'\\'+'tap_fun_train.csv')
    table_test=pd.read_csv(DATAFOLDER+'\\'+'tap_fun_test.csv')
    return table_train,table_test
def gen_train_data(table_train):
    x_train = table_train.avg_online_minutes
    y_train = table_train.prediction_pay_price.sub(table_train.pay_price)
    return x_train,y_train
def gen_test_data(table_test):
    x_test = table_test.avg_online_minutes
    return x_test
def gen_result(table_test,y_test,tj=0):
    tijiao=table_test.loc[:,['user_id','pay_price']]
    tijiao['prediction_pay_price']=y_test   #########可能有误
    tijiao.prediction_pay_price=tijiao.prediction_pay_price.add(tijiao.pay_price)
    del tijiao['pay_price']
    if tj:
        tijiao.to_csv(r'E:\works\python-works\dc_game\tijiao.csv',index=False)
def linear_regression(x_train,y_train,x_test):
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    x_train=x_train.reshape(-1,1)
    x_test=x_test.reshape(-1,1)
    regr = linear_model.LinearRegression()
    regr.fit(x_train, y_train)
    y_test = regr.predict(x_test)
    print('Coefficients:%s, intercept %.2f'%(regr.coef_,regr.intercept_))  #输出权重向量w，和权重b
    return y_test
if __name__=='__main__':
    tstart=t0=time.time()
    table_train,table_test = load_data()
    x_train,y_train = gen_train_data(table_train)
    x_test = gen_test_data(table_test)
    print('Data ready by %.1f seconds!'%(time.time()-t0));t0=time.time()
    y_test = linear_regression(x_train,y_train,x_test)
    print('Data computed by %.1f seconds!'%(time.time()-t0));t0=time.time()
    gen_result(table_test,y_test,tj=1)
    print('Total time is %.1f seconds!'%(time.time()-tstart));