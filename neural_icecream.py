#필요한 라이브러리 불러오기
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

#dataset 불러오기
data = pd.read_csv("icecream_sales.csv")

#x값과 y값 초기화시키기
x_train = data.temp
y_train = data.sales

# keras model 설계
regularizer=tf.keras.regularizers.l2(0.0001) # 문제에서 제시한 정규화 0.0001
inputs = layers.Input(shape = (1,)) # input 차원 설정
x = layers.Dense(1, activation= "relu", kernel_regularizer = regularizer)(inputs) # 문제에서 제시한 activation RELU 및 정규화

outputs = (x) #결과값

model = tf.keras.Model(inputs, outputs) # model 적용

model.compile( # learning rate 0.01적용 후 compile
    loss = tf.keras.losses.mean_squared_error,
    optimizer = tf.keras.optimizers.Adam(0.01)
)

for i in range(0, 25): # complie된 모델에 3일치의 온도로 예측하기 iteration 1000
    model_hist = model.fit(x_train, y_train, epochs=1000)

    model.get_weights()
    temp1 = 26
    predicted_revenue1 = model.predict([temp1])

    temp2 = 24
    predicted_revenue2 = model.predict([temp2])

    temp3 = 39
    predicted_revenue3 = model.predict([temp3])

    print(f"Predicted Revenue: {float(predicted_revenue1): 0.2f}, {float(predicted_revenue2): 0.2f}, {float(predicted_revenue3): 0.2f}")

    file = open("neural_network.txt", 'a') # 텍스트 파일에 저장하기위해
    str = f"Predicted Revenue: {float(predicted_revenue1): 0.2f}, {float(predicted_revenue2): 0.2f}, {float(predicted_revenue3): 0.2f}\n"
    file.write(str)

