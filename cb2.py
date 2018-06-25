# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 18:38:11 2018
用线性模型，对比使用各种特征向量的效果
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
    x_train = table_train[['avg_online_minutes','pay_price','pay_count']]
    x_train['mul_onlinemin_pprice'] = x_train.avg_online_minutes.mul(x_train.pay_price)
    x_train.pay_count = x_train.pay_count.replace(0,0.0001)
    x_train['div_ppric-pcount'] = x_train.pay_price.div(x_train.pay_count)
    del x_train['pay_count']
    y_train = table_train.prediction_pay_price.sub(table_train.pay_price)
    return x_train,y_train
def gen_test_data(table_test):
    x_test = table_test[['avg_online_minutes','pay_price','pay_count']]
    x_test['mul_onlinemin_pprice'] = x_test.avg_online_minutes.mul(x_test.pay_price)
    x_test.pay_count = x_test.pay_count.replace(0,0.0001)
    x_test['div_ppric-pcount'] = x_test.pay_price.div(x_test.pay_count)
    del x_test['pay_count']
    return x_test
def gen_result(table_test,y_test,tj=0):
    tijiao=table_test[['user_id','pay_price']]
    tijiao['prediction_pay_price']=y_test   
    tijiao.prediction_pay_price=tijiao.prediction_pay_price.add(tijiao.pay_price)
    del tijiao['pay_price']
    if tj:
        tijiao.to_csv(r'E:\works\python-works\dc_game\tijiao.csv',index=False)
def linear_regression(x_train,y_train,x_test):
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
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