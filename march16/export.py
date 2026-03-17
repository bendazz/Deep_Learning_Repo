import json
import torch.nn as nn

def export_model(model, path='model.json'):
    layers = []
    act_names = {
        nn.ReLU: 'relu',
        nn.Sigmoid: 'sigmoid',
        nn.Tanh: 'tanh',
        nn.LeakyReLU: 'leaky_relu',
    }

    # Handle a bare nn.Linear (no hidden layers)
    if isinstance(model, nn.Linear):
        layers.append({
            'weights': model.weight.detach().cpu().tolist(),
            'biases': model.bias.detach().cpu().tolist(),
            'activation': 'sigmoid',
        })
    else:
        # Walk children for Sequential or custom modules
        pending = None
        for module in model.children():
            if isinstance(module, nn.Linear):
                if pending:
                    layers.append(pending)
                pending = {
                    'weights': module.weight.detach().cpu().tolist(),
                    'biases': module.bias.detach().cpu().tolist(),
                    'activation': 'linear',
                }
            elif type(module) in act_names:
                if pending:
                    pending['activation'] = act_names[type(module)]
        if pending:
            layers.append(pending)

    with open(path, 'w') as f:
        json.dump({'layers': layers}, f)
    print(f'Exported {len(layers)}-layer model to {path}')

# --- Example usage ---
#
# No hidden layers (bare nn.Linear):
# model = nn.Linear(2, 1)
# ... train with BCEWithLogitsLoss ...
# export_model(model, 'model.json')
#
# With hidden layers (nn.Sequential):
# model = nn.Sequential(
#     nn.Linear(2, 8),
#     nn.ReLU(),
#     nn.Linear(8, 1),
#     nn.Sigmoid(),
# )
# ... train your model ...
# export_model(model, 'model.json')