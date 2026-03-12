import torch  
import torch.nn as nn 
import torch.optim as optim 
import pandas as pd
from export import export_model

data = pd.read_csv('data.csv')
X = torch.tensor(data.drop('y', axis = 1).values).float()
Y = torch.tensor(data['y'].values).float().reshape(-1,1)

model = nn.Sequential(
    nn.Linear(2,32),
    nn.ReLU(),
    nn.Linear(32,32),
    nn.ReLU(),
    nn.Linear(32,1)
)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(),lr = .01)

epochs = 10000

for epoch in range(epochs):
    Yhat = model(X)
    loss = criterion(Yhat,Y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(loss)

export_model(model,'model.json')