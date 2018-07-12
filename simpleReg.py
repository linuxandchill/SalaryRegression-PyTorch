

import torch
import torch.nn as nn
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv('./salaries.csv')

x_temp = dataset.iloc[:, :-1].values
y_temp = dataset.iloc[:, 1:].values

X_train = torch.FloatTensor(x_temp)
Y_train = torch.FloatTensor(y_temp)

#### MODLE ARCHITECTURE #### 

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1,1)
     
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = Model()

loss_func = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

### TRAINING 
for epoch in range(400):
    y_pred = model(X_train)

    loss = loss_func(y_pred, Y_train)
    print(epoch,loss.data.item())
    #print(loss.data[0].item())
    #plt.scatter(epoch, loss.data[0].item(), color='r', s=10, marker='o')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#plt.xlabel("Epochs")
##plt.ylabel("Epochs")
#plt.show()

test_exp = torch.FloatTensor([[6.0]])
print("If u have 6 yrs exp, Salary is:", model(test_exp).data[0][0].item())
