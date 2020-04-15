# Image utils

import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import neural_transfer.config as cfg


style_paths = {'The Starry Night - Van Gogh':os.path.join(cfg.IMG_STYLE_DIR, 'vangogh.jpg'),
               'Mosaic Lady':os.path.join(cfg.IMG_STYLE_DIR, 'mosaic.jpg'),
               'Seated Nude - Picasso':os.path.join(cfg.IMG_STYLE_DIR, 'picasso.png'),
               'The Great Wave off Kanagawa - Hokusai':os.path.join(cfg.IMG_STYLE_DIR, 'tsunami.jpg')}
    

def image_loader(image_name, imsize, heigth, width, device):
    loader = transforms.Compose([transforms.Resize(imsize),  # scale imported image
                                 transforms.ToTensor()])  # transform it into a torch tensor
    image = Image.open(image_name)
    image = image.convert('RGB')
  
    image = image.resize((width, heigth), Image.ANTIALIAS)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)






