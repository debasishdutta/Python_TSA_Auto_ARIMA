#!/usr/bin/env python
# coding: utf-8

# In[1]:


#### USER INPUT SECTION ###

# Global Variables #
global_source_train = "C:/Users/debas/Downloads/Python Code Library/Forecasting Model Code/Datasets/Timeseries_Train_Data.csv"
global_source_test = "C:/Users/debas/Downloads/Python Code Library/Forecasting Model Code/Datasets/Timeseries_Test_Data.csv"
global_ts_column = "TS_Var"
global_date_column = "Date"
global_data_freq = 30                                        # Daily (30)/ Monthly(12)/ Weekly(26)/ Quarterly(4) #

# Model Configurations For ARIMA MODEL
arima_max_p = 2
arima_max_d = 2
arima_max_q = 2
arima_max_P = 2
arima_max_D = 2
arima_max_Q = 2
arima_trend = 'ct'                                           # 'c': Constant, 't': linear Trend, 'ct': Constant + Linear Trend #
arima_information_criterion = 'aic'                          # 'aic' or 'bic'
arima_alpha = 0.05                                           # Prob Cut off For Stationarity Check #
arima_maxiter = 100


# In[2]:


### IMPORT ALL NECCESSARY PACKAGES ###

import numpy as np
import pandas as pd
from pmdarima.arima import auto_arima


# In[3]:


### DATA IMPORT ###

def data_import(source_data, date_col):
    
    df = pd.read_csv(source_data,
                     header = 0,
                     index_col= date_col,
                     parse_dates=True,
                     infer_datetime_format=True,
                     squeeze=True)
    df = df.fillna(method='ffill')
    
    return(df)


# In[6]:


### ARIMA Modelling ###

def model_arima(train,
                test,
                df_x, 
                param_max_p, 
                param_max_d, 
                param_max_q, 
                param_max_P, 
                param_max_D, 
                param_max_Q,
                param_trend,
                param_ic,
                param_p_cutoff,
                param_max_iter,
                param_period):
    
    model_arima = auto_arima(train,
                             exogenous = df_x,                             
                             max_p = param_max_p,
                             max_d = param_max_d,
                             max_q = param_max_q,
                             max_P = param_max_P,
                             max_D = param_max_D,
                             max_Q = param_max_Q,
                             seasonal = True,
                             stationary = False,
                             trend = param_trend,
                             m = param_period,
                             information_criterion = param_ic,
                             alpha = param_p_cutoff,
                             maxiter = param_max_iter,
                             stepwise = True,
                             trace = True, 
                             error_action = 'ignore',
                             suppress_warnings = True)
    
    test_sample_output = pd.DataFrame({"Actuals" : test,
                                       "Predicted" : model_arima.predict(n_periods=len(test))})
    return(test_sample_output)


# In[7]:


ts_train = data_import(global_source_train, global_date_column)
ts_test = data_import(global_source_test, global_date_column)
arima_output = model_arima(ts_train,
                           ts_test,
                           None,
                           arima_max_p,
                           arima_max_d,
                           arima_max_q,
                           arima_max_P,
                           arima_max_D,
                           arima_max_Q,
                           arima_trend,
                           arima_information_criterion,
                           arima_alpha,
                           arima_maxiter,
                           global_data_freq)


# In[ ]:




