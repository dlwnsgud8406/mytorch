#필요한 라이브러리들
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#dataset 가져오기
df = pd.read_csv('icecream_sales.csv')

#dataset으로 부터 값 배열 만들기
x = df.loc[:, 'temp'].values
y = df.loc[:, 'sales'].values

xp = torch.from_numpy(x).float()
yp = torch.from_numpy(y).float()


xp = xp.view(len(xp),1)
yp = yp.view(len(yp),1)

#선형회귀 모델 클래스 생성
class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize) # 선형회귀 layer

    def forward(self, x):
        out = self.linear(x)
        return out

inputDim = 1 # temp의 차원
outputDim = 1  #sales의 차원
learningRate = 0.02 # 학습률
epochs = 1000 # 시행

model = linearRegression(inputDim, outputDim) # 선형모델 선언

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate) #Adam으로 optimization

losses = []
final_loss = 0
for epoch in range(epochs):
    # temp값 입력
    y_pred = model(xp)

    # 결과값 예측하기 위해 loss function실행
    loss = criterion(y_pred, yp)

    losses.append(loss.item())

    #이전값에 영향을 주지 않기위해 기울기를 0으로 하기
    optimizer.zero_grad()

    #이전값에 영향을 주지않기위해
    loss.backward()

    final_loss = loss.item()

    # parameter들 update
    optimizer.step()

    print('epoch: ', epoch, ' , loss: ', loss.item())


y_pred = model(xp)
plt.scatter(xp.cpu(),yp.cpu())
plt.plot(xp.cpu(),y_pred.detach().cpu(),'r')
plt.xlabel('Temp')
plt.ylabel('sales')

#실제 3날짜를 검증하기위한 dataset불러오기
val_df = pd.read_csv('val1.csv')
val_x = val_df.loc[:, 'temp'].values
val_xp = torch.from_numpy(val_x).float()
val_xp = val_xp.view(len(val_xp),1)
val_y_pred = model(val_xp) # 검증하기
print(val_y_pred) # 예측값 출력
