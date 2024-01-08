import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
import matplotlib.pyplot as plt

from .img_dataset import get_emnist, get_kmnist
from .cae import ConvDecoder


class VarConvEncoder(nn.Module):
    def __init__(self, hid_units=200):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(2)
        self.feats2hid = nn.Linear(64*8*8, hid_units) # 4096 -> 200 hidden units
        
        self.hid2mu = nn.Linear(hid_units, 8) # 200 -> 8
        self.hid2sigma = nn.Linear(hid_units, 8) # 200 -> 8
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.feats2hid(x))
        
        mu = self.hid2mu(x)
        sigma = self.hid2sigma(x)
        return mu, sigma


class VarConvAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # N, 1, 32, 32
        self.encoder = VarConvEncoder()
        self.decoder = ConvDecoder()
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        mu, sigma = self.encoder(x)
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        x = self.decoder(z)
        return x, mu, sigma


def train_vcae(dl, criterion, num_epochs=2, viz=False, name='', lossname='bce', device='cpu'):
    # Check if gpu is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = criterion.to(device)
    model = VarConvAutoEncoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_loss = float('inf')
    best_epoch = -1
    best_model = None
    best_outputs = []
    for epoch in range(num_epochs):
        cum_loss = 0
        for batch_id, img in enumerate(dl):
            img = img.to(device)
            recon, mu, sigma = model(img)

            recon_loss = criterion(recon, img) # 1 - criterion(recon, img) if SSIM else criterion(recon, img)
            kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
            loss = recon_loss + kl_div
            cum_loss += loss

            if batch_id == 0: # Save batch for visualizations
                output = (epoch, batch_id, img.cpu(), recon.cpu())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch: {epoch}, Loss: {cum_loss.item():.4f}')

        if cum_loss < best_loss:
            best_loss = cum_loss
            best_model = model.state_dict()
            best_epoch = epoch
        best_outputs.append(output)

    if viz:
        visualize_outputs(best_outputs, name, lossname)

    # Save the best model to a file
    print(f'Save best model from epoch {best_epoch} with loss {best_loss}')
    torch.save(best_model, f"vcae_mse/{name + '_' if name else ''}vcae_{lossname}_best_{best_epoch}.pth")


def visualize_outputs(outputs, name='mae_emnist', lossname='bce'):
    for output in outputs:
        vizualize_batch((output[2], output[3]), name=name, lossname=lossname, epoch=output[0], batch_id=output[1])


def vizualize_batch(output, name, epoch, batch_id, lossname='bce'):
    n = output[0].shape[0]
    plt.figure(figsize=(2, n))
    plt.gray()
    imgs = output[0].detach().numpy()
    recons = output[1].detach().numpy()

    for i in range(len(imgs)):
        plt.subplot(n, 2, i*2+1)
        plt.imshow(imgs[i][0])
        plt.subplot(n, 2, i*2+2)
        plt.imshow(recons[i][0])
    plt.savefig(f'vcae_mse/{name}/{name}_{epoch}_{batch_id}_{lossname}.png')
    plt.close()


if __name__ == '__main__':
    criterion = nn.MSELoss(reduction='sum') #nn.MSELoss() StructuralSimilarityIndexMeasure(data_range=1) 
    dl = get_emnist()
    train_vcae(dl, criterion, name='emnist', lossname='mse', viz=True)

    dl = get_kmnist()
    train_vcae(dl, criterion, name='kmnist', lossname='mse', viz=True)
