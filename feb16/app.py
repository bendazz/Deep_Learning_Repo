import pandas as pd  
import numpy as np 
import torch  
import torch.nn as nn  
import torch.optim as optim  

data = pd.read_csv('data.csv')

features = torch.tensor(data.drop('Price',axis = 1).to_numpy()).float()
target = torch.tensor(data['Price'].to_numpy()).float().reshape(-1,1)

fm = features.mean().reshape(-1,1)
fs = features.std().reshape(-1,1)
tm = target.mean().reshape(-1,1)
ts = target.std().reshape(-1,1)

X = (features - fm)/fs
Y = (target - tm)/ts

model = nn.Linear(1,1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr = .1)

epochs = 100
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

