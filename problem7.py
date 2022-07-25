
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas
import matplotlib as plt
import numpy

torch.manual_seed(1)
train_dataset = pandas.read_csv('/Users/ijunhyeong/Desktop/mytorch/icecream_sales.csv')

x = train_dataset.loc[:, 'temp']
y = train_dataset.loc[:, 'sales']

xp = torch.from_numpy(x).float().cuda()
yp = torch.from_numpy(y).float().cuda()

class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

inputDim = 1        # takes variable 'x'
outputDim = 1       # takes variable 'y'
learningRate = 0.02
epochs = 25

model = linearRegression(inputDim, outputDim).cuda()


criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

losses = []
final_loss = 0
for epoch in range(epochs):
    # get output from the model, given the inputs
    y_pred = model(xp)

    # get loss for the predicted output
    loss = criterion(y_pred, yp)

    losses.append(loss.item())

    # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
    optimizer.zero_grad()

    # get gradients w.r.t to parameters
    loss.backward()

    final_loss = loss.item()

    # update parameters
    optimizer.step()

    print('epoch: ', epoch, ' , loss: ', loss.item())

y_pred = model(xp)

plt.scatter(xp.cpu(),yp.cpu())
plt.plot(xp.cpu(),y_pred.detach().cpu(),'r')
plt.xlabel('experience')
plt.ylabel('salary')
