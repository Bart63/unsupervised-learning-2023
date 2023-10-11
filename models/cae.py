import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from img_dataset import get_emnist, get_kmnist

class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # N, 1, 32, 32
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1), # -> N, 8, 16, 16
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), # -> N, 16, 8, 8
            nn.ReLU(),
            nn.Conv2d(16, 32, 8), # -> N, 32, 1, 1
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 8), # -> N, 16, 8, 8
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1), # -> N, 8, 16, 16
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1), # -> N, 1, 32, 32
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    # Check if gpu is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvAutoEncoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # dl = get_emnist()
    dl = get_kmnist()

    num_epochs = 50
    best_loss = float('inf')
    best_epoch = -1
    best_model = None
    outputs = []
    for epoch in range(num_epochs):
        for img in dl:
            img = img.to(device)
            recon = model(img)
            loss = criterion(recon, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')

        if loss < best_loss:
            best_loss = loss
            best_model = model.state_dict()
            best_epoch = epoch

        outputs.append((epoch, img.cpu(), recon.cpu()))

    # Save the best model to a file
    torch.save(best_model, f'cae_best_{best_epoch}.pth')

    plt.figure(figsize=(18, 2))
    plt.gray()
    imgs = outputs[-1][1].detach().numpy()
    recon = outputs[-1][2].detach().numpy()
    for i, item in enumerate(imgs):
        if i >= 18: break
        plt.subplot(2, 18, i+1)
        plt.imshow(item[0])
            
    for i, item in enumerate(recon):
        if i >= 18: break
        plt.subplot(2, 18, 18+i+1)
        plt.imshow(item[0])
    plt.show()