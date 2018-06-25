# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 21:27:31 2018
选择验证集，对预测效果自测
@author: 陈搏
"""

import pandas as pd
import cb2

DATAFOLDER=r'E:\works\python-works\dc_game\data'
def gen_validation_data():
    table = pd.read_csv(DATAFOLDER+'\\'+'tap_fun_train.csv')
    nTest = 500000
    
    table_test = table.sample(n=nTest)
    index_test=table.user_id.isin(table_test.user_id)
    index_train = index_test.apply(lambda x: not x)
    table_train = table[index_train]
    
    table_train.to_csv(DATAFOLDER+'\\'+'validation_train.csv',index=False)
    table_test.to_csv(DATAFOLDER+'\\'+'validation_test.csv',index=False)
def load_validation_data():
    table_train = pd.read_csv(DATAFOLDER+'\\'+'validation_train.csv')
    table_test = pd.read_csv(DATAFOLDER+'\\'+'validation_test.csv')
    return table_train,table_test
def RMSE_cal(all_pay_pred,all_pay_real):
    RMSE=((all_pay_pred.sub(all_pay_real).apply(lambda x: x**2).sum())/len(all_pay_pred))**0.5
    print('RMSE in validation data is %.2f.'%RMSE)
    return RMSE
if __name__=='__main__':
    table_train,table_test = load_validation_data()
    x_train,y_train = cb2.gen_train_data(table_train)
    x_test = cb2.gen_test_data(table_test)
    y_test = cb2.linear_regression(x_train,y_train,x_test)
    all_pay_pred = table_test.pay_price.add(y_test)
    all_pay_real = table_test.prediction_pay_price
    RMSE = RMSE_cal(all_pay_pred,all_pay_real)