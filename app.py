import os
import cv2
import torch
from ml import ConvAutoEncoder


DATASET_FOLDER = 'dataset'
MODELS_FOLDER = 'models'

class EncoderModel:
    def __init__(self, ds='emnist'):
        self.ds = ds
        self.load_models()

    def load_models(self, type='cae'):
        sd_path = os.path.join(MODELS_FOLDER, f'{self.ds}_{type}.pth')
        self.model:ConvAutoEncoder = ConvAutoEncoder()
        self.model.load_state_dict(torch.load(sd_path))
    
    def encode(self, img):
        return self.model.encoder(img) 


class PageProcessor:
    def __init__(self) -> None:
        self.emnist_model = EncoderModel('emnist')
        self.kmnist_model = EncoderModel('kmnist')
        self.load_pages()

    def load_pages(self):
        emnist_path = os.path.join(DATASET_FOLDER, 'emnist_page.png')
        self.emnist_page = cv2.imread(emnist_path, cv2.IMREAD_GRAYSCALE)
        kmnist_path = os.path.join(DATASET_FOLDER, 'kmnist_page.png')
        self.kmnist_page = cv2.imread(kmnist_path, cv2.IMREAD_GRAYSCALE)
    
    def encode_first(self): # Test method to check if everything is working
        size = 32
        img = self.emnist_page[:size, :size]
        tensor = torch.from_numpy(img.reshape(1, 1, size, size)).float()
        return img, self.emnist_model.encode(tensor).detach().numpy()
    
    def emnist_encode(self, img):
        return self.emnist_model.encode(img)

if __name__ == '__main__':
    page_processor = PageProcessor()
    img, rep = page_processor.encode_first()
    cv2.imshow('character img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(rep)
