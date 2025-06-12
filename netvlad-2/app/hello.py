import torch

checkpoint = torch.load('../vgg16_netvlad_checkpoint.pth.tar', map_location='cpu', weights_only=False)
print("Checkpoint keys:", checkpoint.keys())
