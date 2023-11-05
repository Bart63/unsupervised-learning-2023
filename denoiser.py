from ml import ConvAutoEncoder
import os
import torch

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

MODELS_FOLDER = 'models'


class Denoiser:
    def __init__(self, ds='emnist'):
        self.ds = ds
        self.load_models()

    def load_models(self, type='cae'):
        sd_path = os.path.join(MODELS_FOLDER, f'{self.ds}_{type}.pth')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model:ConvAutoEncoder = ConvAutoEncoder()
        self.model = ConvAutoEncoder().to(device)
        # self.model.load_state_dict(torch.load(sd_path))
        state_dict = torch.load(sd_path, map_location=device)
        self.model.load_state_dict(state_dict)

    def denoise(self, img):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor = torch.from_numpy(img.reshape(1, 1, 32, 32)).float().to(device)
        with torch.no_grad():
            img = self.model.encoder(tensor)
            img = self.model.decoder(img)
        return img


# test
'''
denoiser = Denoiser()
path = 'dataset/pairs_e22_k2_s0/0.png'
image = Image.open(path)
width, height = image.size
image = image.crop((0, 0, width // 2, height))
image = np.array(image)
plt.imshow(image, cmap='gray')
plt.axis('off') 
plt.show()

image = denoiser.denoise(image)
image = image.reshape(32, 32)
plt.imshow(image, cmap='gray')
plt.axis('off') 
plt.show()
'''
