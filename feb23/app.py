import torch
import torch.nn as nn

model_data = torch.load('model.pth')
fm = model_data['fm']
fs = model_data['fs']
parameters = model_data['parameters']

linear = nn.Linear(1,1)
linear.load_state_dict(parameters)

model = nn.Sequential(
    linear,
    nn.Sigmoid()
)

features = torch.tensor([
    [3.3]
])

X = (features - fm)/fs

prob = model(X)

if prob > .5:
    classification = "Malignant"
else:
    classification = "Benign"

print(prob, classification)


