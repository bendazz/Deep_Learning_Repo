import torch
import torch.nn as nn

model_data = torch.load('model.pth')
fm = model_data['fm']
fs = model_data['fs']
tm = model_data['tm']
ts = model_data['ts']
parameters = model_data['parameters']


features = torch.tensor([
   [3.0,2.0] 
])

X = (features-fm)/fs

model = nn.Linear(2,1)
model.load_state_dict(parameters)

prediction  = model(X)
print(prediction*ts + tm)