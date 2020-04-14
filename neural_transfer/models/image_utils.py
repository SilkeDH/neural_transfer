# Image utils

import torch
from PIL import Image
import torchvision.transforms as transforms

def image_loader(image_name, imsize, heigth, width, device):
    loader = transforms.Compose([transforms.Resize(imsize),  # scale imported image
                                 transforms.ToTensor()])  # transform it into a torch tensor
    image = Image.open(image_name)
    image = image.convert('RGB')
  
    image = image.resize((width, heigth), Image.ANTIALIAS)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)




