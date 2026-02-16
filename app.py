import torch
import torch.nn as nn
import gradio as gr   

model_data = torch.load('model.pth')
model = nn.Linear(1,1)
model.load_state_dict(model_data['model_state_dict'])



def predict(input):
    if input != None:
        feet = torch.tensor([
            [input]
        ])
        X = (feet - model_data['feet_mean'])/model_data['feet_std']
        Yhat  = model(X)
        price = Yhat*model_data['price_std'] + model_data['price_mean']
        return price.item()
    else:
        return None

with gr.Blocks() as iface:
    feet_box = gr.Number(label = 'type in a square footage')
    price_box = gr.Number(label = 'the predicted price in thousands is:')
    feet_box.change(fn = predict,inputs = [feet_box],outputs = [price_box])

iface.launch()