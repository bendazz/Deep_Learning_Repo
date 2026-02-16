import torch
import torch.nn as nn
import gradio as gr  


model_data = torch.load('model.pth')


fm = model_data['fm']
fs = model_data['fs']
tm = model_data['tm']
ts = model_data['ts']
parameters = model_data['model_state_dict']

model = nn.Linear(1,1)
model.load_state_dict(parameters)


def predict(feet):
    features = torch.tensor([
        [feet]
    ]).float()
    X = (features - fm)/fs
    Yhat = model(X)
    Price = Yhat*ts+tm
    return Price.item()

with gr.Blocks() as iface:
    feet_box = gr.Number(label= "type in feet")
    price_box = gr.Number(label= "this is the predicted price")
    feet_box.change(fn = predict,inputs = [feet_box],outputs = [price_box])
    
iface.launch()