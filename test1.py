import torch
weight_path = 'weights/pretrained/mobilenetv2.pt'

state_dict = torch.load(weight_path, map_location=0)
print(state_dict)