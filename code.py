#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import math
import talib
import os
from hmmlearn import hmm


# In[3]:


def data_preprocess():
    '''
    preprocess the data: calculate the input data of model
    add the three column into the original dataFrame
    '''
    #获取文件名
    file_names = os.listdir('Data')
    pre_path = 'Data/'
    
    # 更新dataFrame
    for file_name in file_names:
        path = pre_path + file_name
        data = pd.read_pickle(path)
        data['close_open'] = (data['close'] - data['open'])/data['open']
        data['high_open'] = (data['high'] - data['open'])/data['open']
        data['open_low'] = (data['open'] - data['low'])/data['open']
        data = data[data['paused']==0]
        data = data[(data['close_open']<0.1) & (data['close_open']> -0.1)]
        data = data[(data['high_open']<0.1)]
        data = data[(data['open_low']<0.1)]
        data.to_pickle(path)
        print(path + '处理完成')


# In[4]:


def get_stock_data(code):
    '''
    get the train data
    input: the code of the stock
    output: return the data which do not have the pause time
    '''
    pre_path = 'Data/'
    path = pre_path + code + '.pkl'
    
    data = pd.read_pickle(path)
    
    pro_data = data[data['paused']==0]
    return pro_data

def get_model_input(code):
    '''
    get the input of model
    input: the code of stock
    output:np.array
    '''
    data = get_stock_data(code)
    input_data = data.loc[:, ['close_open', 'high_open', 'open_low']].iloc[:(len(data)-200)].values
    
    return input_data


# In[11]:


def create_model(n_components, n_mix):
    model = hmm.GMMHMM(n_components=n_components, n_mix=n_mix, init_params='mcw')
    start_probability = np.ones(n_components)
    start_probability = start_probability/n_components
    transition_probability = np.ones((n_components,n_components))
    transition_probability = transition_probability/n_components
    
    model.startprob_ = start_probability
    model.transmat_ = transition_probability
    
    return model

def train_model(model, code):
    
    '''file_names = os.listdir('Data')
    is_begin = True
    
    for file_name in file_names:
        file_name = file_name[:-4]
        if is_begin:
            input_data = get_model_input(file_name)
            is_begin = False
        else:
            input_data = np.concatenate((input_data, get_model_input(file_name)), axis=0)
        print(len(input_data))
        print(file_name)
    model = model.fit(input_data)
    return model'''
    input_data = get_model_input(code)
    model = model.fit(input_data)
    return model
    
def get_model(code):
    '''
    get the GMMHMM model according to the input
    '''
    n_components = 4
    n_mix = 5
    
    model = create_model(n_components, n_mix)
    model = train_model(model, code)
    
    return model

def predict_model(model, code):
    '''
    predict the state sequence and save the last
    '''
    test_data = get_stock_data(code)
    if len(test_data) < 100:
        state_num = [math.nan]*len(test_data)
    else:
        state_num = [math.nan]*9
        for i in range(len(test_data)-9):
            state_seq = model.predict(test_data.loc[:, ['close_open', 'high_open', 'open_low']].iloc[:i+10].values)
            state_num.append(state_seq[-1])
    
    # write into file
    path = 'Data/' + code + '.pkl'
    data = pd.read_pickle(path)
    data['state'] = state_num
    data.to_pickle(path)
    


# In[6]:


from scipy.stats import multivariate_normal

def calculate_best(model, state_num):
    '''
    calculate the best predict
    input1: the trained GMMHMM model
    input2: the last state_num
    output: return the best ((close-open)/open)
    '''
    state_num = int(state_num)
    state_pro = model.transmat_[state_num]
    next_state = np.where(state_pro == np.max(state_pro))[0][0]
    
    normal_list = create_normal_model(model, next_state)
    best_value = 0
    best_pro = 0
    
    is_modify = False
    
    for i in range(50):
        is_modify = False
        first = (-100 + 200/50*(i+1))/1000
        for j in range(10):
            second = 0.1/10 *(j+1)
            for k in range(10):
                third = 0.1/10 *(k+1)
                pro = cal_sigle_value(normal_list, first, second, third, model, next_state)
                
                if pro > best_pro:
                    best_pro = pro
                    best_value = first
                    is_modify = True
                    break
            
            if is_modify:
                break
    
    return best_value                
    
def create_normal_model(model, next_state):
    '''
    create the normal model according to the next_state
    '''
    normal_list = []
    for i in range(5):
        normal_model = multivariate_normal(mean=model.means_[next_state][i], cov=model.covars_[next_state][i])
        normal_list.append(normal_model)
    return normal_list


def cal_sigle_value(normal_list, first, second, third, model, next_state):
    '''
    calculate the probility of the value
    '''
    values = []
    for i in range(5):
        normal_model = normal_list[i]
        values.append(normal_model.pdf([first, second, third]))
    
    np_values = np.array(values)
    return np.dot(np_values, model.weights_[next_state].T)


def cal_all_close(model, code):
    '''
    calculate the close and write into file
    '''
    path = 'Data/' + code + '.pkl'
    data = pd.read_pickle(path)
    
    if (len(data) < 100):
        pre_close_open = [math.nan]*len(data)
        pre_close = [math.nan]*len(data)
    else:
        state_num = data['state']
        pre_close_open = [math.nan]*9
        pre_close = [math.nan]*9
        
        predict_values = []
        for state in state_num:
            if math.isnan(state):
                continue    
            predict_value = calculate_best(model, state)
            predict_values.append(predict_value)
        
        np_predict_value = np.array(predict_values)
        open_values = data['open'].iloc[9:].values
        np_predict_close = np_predict_value*open_values + open_values
        
        pre_close_open = np.concatenate((pre_close_open, np_predict_value), axis=0)
        pre_close = np.concatenate((pre_close, np_predict_close), axis=0)
        
    data['pre_close_open'] = pre_close_open
    data['pre_close'] = pre_close
    
    data.to_pickle(path)


# In[7]:


def save_model(model, code):
    '''
    save the model into pkl file
    '''
    pre_path = 'model/'
    path = pre_path + 'pro_'+ code + '.pkl'
    
    data = pd.DataFrame()
    data['start_0'] = model.startprob_[0]
    data['transition_0'] = model.transmat_[0]
    data['transition_1'] = model.transmat_[1]
    data['transition_2'] = model.transmat_[2]
    data['transition_3'] = model.transmat_[3]
    data.to_pickle(path)
    
    pre_path = 'model/'
    path = pre_path + 'weights_'+ code + '.pkl'
    data = pd.DataFrame(model.weights_)
    data.to_pickle(path)
    
    pre_path = 'model/'
    path = pre_path + 'means_'+ code + '.pkl'
    all_means = model.means_[0]
    all_means = np.concatenate((all_means, model.means_[1]), axis=0)
    all_means = np.concatenate((all_means, model.means_[2]), axis=0)
    all_means = np.concatenate((all_means, model.means_[3]), axis=0)
    data = pd.DataFrame(all_means)
    data.to_pickle(path)
    
    pre_path = 'model/'
    path = pre_path + 'coval_'+ code + '.pkl'
    all_cov = model.covars_[0]
    all_cov = np.concatenate((all_cov, model.covars_[1]), axis=0)
    all_cov = np.concatenate((all_cov, model.covars_[2]), axis=0)
    all_cov = np.concatenate((all_cov, model.covars_[3]), axis=0)
    data = pd.DataFrame(all_cov)
    data.to_pickle(path)
    


# In[12]:


file_names = os.listdir('Data')
pre_path = 'Data/'

for file_name in file_names:
    if os.path.exists('model/pro_'+file_name):
        print(file_name+':the model is exist')
        continue
    
    code = file_name[:-4]
    
    if len(get_stock_data(code)) <= 500:
        print(code + ": length is not enough")
        continue
        
    model = get_model(code)
    predict_model(model, code)
    cal_all_close(model, code)
    save_model(model, code)
    print(code + ': Successful')


# In[26]:


def show_result(code):
    path = 'Data/'+code+'.pkl'
    data = pd.read_pickle(path)
    plt.figure(figsize=(20,10))
    plt.plot(data['close'].iloc[-200:], label='the real price')
    plt.plot(data['pre_close'].iloc[-200:], label='the predict price')
    plt.legend(loc=0)
    #plt.show()
    plt.savefig("price.png")
    
    plt.figure(figsize=(20,10))
    for i in range(4):
        show_data = data.iloc[-200:][data['state'] == i]
        plt.plot(show_data['close'],'.', label = 'hidden_state %d'%i)
    plt.plot(data['close'].iloc[-200:], color='lightgray', zorder=0.0)
    plt.legend(loc=0)
    #plt.show()
    plt.savefig("state.png")
show_result('000008.XSHE')


# In[73]:


def cal_mape():
    file_names = os.listdir('Data')
    pre_path = 'Data/'
    
    results = pd.DataFrame(columns=['code', 'MAPE'])
    
    for file_name in file_names:

        code = file_name[:-4]
        data = get_stock_data(code)
        
        if len(data) <= 500:
            print(code + ": length is not enough")
            continue
        
        result = (abs((data['pre_close'].iloc[-200:] - data['close'].iloc[-200:]))/data['close'].iloc[-200:]).sum()
        result = result/200
        results.loc[len(results)] = [code, result]
        print(code + ': Successful')
        
    results.set_index('code', inplace=True)
    path = 'Result/results.csv'
    results.to_csv(path)

cal_mape()
        
        
        


# In[ ]:




