import torch 
import torch.nn as nn 
import torch.optim as optim 
import pandas as pd 
import numpy as np 

torch.manual_seed(1)

data = pd.read_csv('data.csv')
features = torch.tensor(data.drop('MPG',axis = 1).to_numpy()).float()
target = torch.tensor(data['MPG'].to_numpy()).float().reshape(-1,1)

fm = features.mean(axis = 0,keepdim = True)
fs = features.std(axis = 0,keepdim = True)
tm = target.mean(axis = 0,keepdim = True)
ts = target.std(axis = 0,keepdim = True)

X = (features - fm)/fs
Y = (target - tm)/ts


model = nn.Linear(2,1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr = .1)

epochs = 10000
for epoch in range(epochs):
    Yhat = model(X)
    loss = criterion(Yhat,Y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    

torch.save({
    'fm':fm,
    'fs':fs,
    'tm':tm,
    'ts':ts,
    'parameters':model.state_dict()
},'model.pth')