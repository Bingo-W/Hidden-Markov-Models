#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import math
import talib
import os
from hmmlearn import hmm


# In[75]:


file_names = os.listdir('Data')
pre_path = 'Data/'

profile_result = pd.DataFrame(columns=['code', 'buy_time', 'buy_price', 'sell_time', 'sell_price'])
for file_name in file_names:
    
    data = pd.read_pickle(pre_path + file_name)
    if len(data) <= 500:
        print(code + ": length is not enough")
        continue
    
    code = file_name[:-4]
    data = data.iloc[-200:]
    
    mean_path = 'model/'+'means_'+file_name
    mean_data = pd.read_pickle(mean_path)
    buy_state = 0
    sell_state = 0
    temp_max = 0
    temp_min = 0
    for i in range(4):
        temp_mean = mean_data[0].iloc[i*0:i+5].mean()
        if temp_mean > temp_max:
            temp_max = temp_mean
            buy_state = i
            continue
        elif temp_mean < temp_min:
            temp_min = temp_mean
            sell_state = i
            continue

    is_long = False
    result = []
    
    day_limit = 1
    day = 0
    if temp_max < 0.04:
        print(code + ': max is too small')
        continue
        
    for i in range(len(data)):
        temp_data = data.iloc[i]
        day += 1
        if is_long:
            if temp_data['state'] == sell_state:
                is_long = False
                result.append(data.index[i])
                result.append(temp_data['open'])
                profile_result.loc[len(profile_result)] = result
                result = []
                continue
            else:
                if day >= day_limit:
                    is_long = False
                    result.append(data.index[i])
                    result.append(temp_data['open'])
                    profile_result.loc[len(profile_result)] = result
                    result = []
                else:
                    continue
        else:
            if temp_data['state'] == buy_state:
                is_long = True
                day = 0
                result.append(code)
                result.append(data.index[i])
                result.append(temp_data['open'])
                continue
            else:
                continue
    print(code + ': Successfully')

profile_result['profile'] = (profile_result['sell_price']-profile_result['buy_price'])/profile_result['buy_price']
profile_result.to_csv('profile_have.csv')


# In[94]:


profile_result = pd.read_csv('profile_have.csv')
np_profit = np.array(profile_result['profile'])

print('交易信号次数：'+ str(len(np_profit)))
print('平均值：', np.mean(np_profit))
print('胜率：', str(len(profile_result[profile_result['profile']>0])/len(profile_result)))
print('收益波动', np.std(np_profit))
print('中位值：', np.median(np_profit))
print('90%分位数：', np.percentile(np_profit, 90))
print('80%分位数：', np.percentile(np_profit, 80))
print('70%分位数：', np.percentile(np_profit, 70))
print('60%分位数：', np.percentile(np_profit, 60))
print('50%分位数：', np.percentile(np_profit, 50))
print('40%分位数：', np.percentile(np_profit, 40))
print('30%分位数：', np.percentile(np_profit, 30))
print('20%分位数：', np.percentile(np_profit, 20))
print('10%分位数：', np.percentile(np_profit, 10))

plt.figure(figsize=(20,10))
plt.hist(np_profit, bins=200)
#plt.show()
plt.savefig("2.png")

