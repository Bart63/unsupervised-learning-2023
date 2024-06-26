# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
import matplotlib.pyplot as plt

from .img_dataset import get_emnist, get_kmnist


class ConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(2)
        self.fc = nn.Linear(64*8*8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ConvDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 64*8*8)
        self.conv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = x.view(x.size(0), 64, 8, 8)
        x = F.relu(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        return x


class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # N, 1, 32, 32
        self.encoder = ConvEncoder()
        self.decoder = ConvDecoder()
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# %%

def train_cae(dl, criterion, num_epochs=100, viz=False, name='', lossname='bce', device='cpu'):
    # Check if gpu is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = criterion.to(device)
    model = ConvAutoEncoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_loss = float('inf')
    best_epoch = -1
    best_model = None
    best_outputs = []
    for epoch in range(num_epochs):
        cum_loss = 0
        for batch_id, img in enumerate(dl):
            img = img.to(device)
            recon = model(img)
            loss = 1 - criterion(recon, img) # 1 - criterion(recon, img) if SSIM else criterion(recon, img)
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
    torch.save(best_model, f"cae_ssim/{name + '_' if name else ''}cae_{lossname}_best_{best_epoch}.pth")


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
    plt.savefig(f'cae_ssim/{name}/{name}_{epoch}_{batch_id}_{lossname}.png')
    plt.close()


if __name__ == '__main__':
    criterion = StructuralSimilarityIndexMeasure(data_range=1) #nn.MSELoss() StructuralSimilarityIndexMeasure(data_range=1) 
    dl = get_emnist()
    train_cae(dl, criterion, name='emnist', lossname='ssim', viz=True)

    dl = get_kmnist()
    train_cae(dl, criterion, name='kmnist', lossname='ssim', viz=True)

# %%
