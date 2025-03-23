import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

def load_trained_model(model_path='byol_o3.pth', device='cpu'):
    model = models.resnet18().to(device)
    state_dict = torch.load(model_path, map_location=torch.device(device))
    new_state_dict = {k.replace("online_encoder.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model

def main():
    device = 'cpu'
    model = load_trained_model(device=device)
    np_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    torch_input = torch.from_numpy(np_input).to(device)
    output = model(torch_input)
    print("Model output:", output)

if __name__ == '__main__':
    main()

