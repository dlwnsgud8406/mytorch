import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('icecream_sales.csv')
val_df = pd.read_csv('val1.csv')

df = df.iloc[:, 1:2].values

sc = MinMaxScaler()
training_data = sc.fit_transform(df)

def sliding_windows(data, seq_length):
    x=[]
    y=[]

    for i in range(len(data)-seq_length -1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)
    return np.array(x), np.array(y)
seq_length = 4
x, y = sliding_windows(training_data, seq_length)


# x = df.loc[:, 'temp'].values
# y = df.loc[:, 'sales'].values

xp = torch.from_numpy(x).float()
yp = torch.from_numpy(y).float()

# 7일간의 데이터가 입력으로 들어가고 batch size는 임의로 지정
seq_length = 7
batch = 100

# 데이터를 역순으로 정렬하여 전체 데이터의 70% 학습, 30% 테스트에 사용
df = df[::-1]
train_size = int(len(df)*0.7)
train_set = df[0:train_size]
test_set = df[train_size-seq_length:]


