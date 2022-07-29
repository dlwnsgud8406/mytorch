

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

data = pd.read_csv("icecream_sales.csv")


x_train = data.temp
y_train = data.sales

# model building
regularizer=tf.keras.regularizers.l2(0.0001)
inputs = layers.Input(shape = (1,))
x = layers.Dense(1, activation= "relu", kernel_regularizer = regularizer)(inputs)

outputs = (x)

model = tf.keras.Model(inputs, outputs)


# compiling the model

model.compile(
    loss = tf.keras.losses.mean_squared_error,
    optimizer = tf.keras.optimizers.Adam(0.01)
)

# fitting the data to model


for i in range(0, 25):
    model_hist = model.fit(x_train, y_train, epochs=1000)

    model.get_weights()
    temp1 = 26
    predicted_revenue1 = model.predict([temp1])

    temp2 = 24
    predicted_revenue2 = model.predict([temp2])

    temp3 = 39
    predicted_revenue3 = model.predict([temp3])

    print(f"Predicted Revenue: {float(predicted_revenue1): 0.2f}, {float(predicted_revenue2): 0.2f}, {float(predicted_revenue3): 0.2f}")

    file = open("neural_network.txt", 'a')
    str = f"Predicted Revenue: {float(predicted_revenue1): 0.2f}, {float(predicted_revenue2): 0.2f}, {float(predicted_revenue3): 0.2f}\n"
    file.write(str)

