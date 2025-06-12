import torch
from torchvision import transforms
from PIL import Image
import os
import io
import sys
import numpy

from torch.serialization import safe_globals

sys.path.append(os.path.abspath('../pytorch-NetVlad'))
from netvlad import NetVLAD
from torchvision.models import vgg16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_netvlad_model():
    encoder = vgg16(weights="IMAGENET1K_V1")
    layers = list(encoder.features.children())[:-2]
    encoder = torch.nn.Sequential(*layers)

    net_vlad = NetVLAD(num_clusters=64, dim=512)

    model = torch.nn.Module()
    model.add_module('encoder', encoder)
    model.add_module('pool', net_vlad)

    checkpoint_path = '../vgg16_netvlad_checkpoint.pth.tar'

    with safe_globals([numpy.core.multiarray.scalar]):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    return model

model = load_netvlad_model()

def extract_descriptor(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encoder(image)
        vlad_descriptor = model.pool(features)
        vlad_descriptor = vlad_descriptor.view(-1)
        vlad_descriptor = torch.nn.functional.normalize(vlad_descriptor, p=2, dim=0)
    return vlad_descriptor.cpu()
