import gradio as gr  

def f(x):
    return x**2
    
with gr.Blocks() as iface:
    x_box = gr.Number(label = 'type in a number')
    square_box = gr.Number(label = 'this is the square of that number')
    x_box.change(fn = f,inputs = [x_box],outputs = [square_box])
iface.launch()