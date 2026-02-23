import torch
import torch.nn as nn 
import torch.optim as optim  
import pandas as pd 
import numpy as np

torch.manual_seed(42)

data = pd.read_csv('data.csv')
data['Diagnosis'] = data['Diagnosis'].map({'Benign':0,'Malignant':1})

features = torch.tensor(data.drop('Diagnosis',axis = 1).to_numpy()).float()
Y = torch.tensor(data['Diagnosis'].to_numpy()).reshape(-1,1).float()

fm = features.mean().reshape(-1,1)
fs = features.std().reshape(-1,1)

X = (features - fm)/fs

model = nn.Linear(1,1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(),lr = .1)

epochs = 250
for epoch in range(epochs):
    Yhat = model(X)
    loss = criterion(Yhat,Y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()



torch.save({
    'fm':fm,
    'fs':fs,
    'parameters':model.state_dict()
},'model.pth')
