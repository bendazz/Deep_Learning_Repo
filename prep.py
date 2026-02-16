import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

data = pd.read_csv('data.csv')

#feet_mean = 1600
#price_mean = 240

#feet_std = 432.049380
#price_mean = 64.807407

df_stand = (data - data.mean())/data.std()

X = torch.tensor(df_stand['Feet'].to_numpy().reshape(-1,1)).float()
Y = torch.tensor(df_stand['Price'].to_numpy().reshape(-1,1)).float()

model = nn.Linear(1,1)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr = .01)

epochs = 1000

for epoch in range(epochs):
    Yhat = model(X)
    loss = criterion(Yhat,Y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    

torch.save(
    {
        'model_state_dict':model.state_dict(),
        'feet_mean' : torch.tensor([[1600.0]]),
        'price_mean' : torch.tensor([[240.0]]),
        'feet_std' : torch.tensor([[432.049380]]),
        'price_std' :torch.tensor([[64.807407]]),

    }, 'model.pth'
)