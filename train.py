import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd  
import numpy as np   

data = pd.read_csv('data.csv')
features = torch.tensor(data.drop('Price',axis = 1).to_numpy()).float()
target = torch.tensor(data['Price'].to_numpy()).reshape(-1,1).float()

fm = features.mean(axis = 0)
fs = features.std(axis = 0)
tm = target.mean(axis = 0)
ts = target.std(axis = 0)

X = (features - fm)/fs
Y = (target - tm)/ts

model = nn.Linear(1,1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr = .1)

epochs = 1000

for epoch in range(epochs):
    Yhat = model(X)
    loss = criterion(Yhat,Y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch%10 == 0:
        print(loss.item())


torch.save({
    'fm':fm,
    'fs':fs,
    'tm':tm,
    'ts':ts,
    'model_state_dict':model.state_dict()
},'model.pth')


