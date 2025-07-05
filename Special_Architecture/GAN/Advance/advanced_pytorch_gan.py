import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE_G = 0.0002
LEARNING_RATE_D = 0.0002
BETA1 = 0.5
BETA2 = 0.999

LATENT_DIM = 100
IMAGE_SIZE = 64
NUM_CHANNELS = 1
FEATURE_MAP_G = 64
FEATURE_MAP_D = 64

LABEL_SMOOTHING = 0.1
NOISE_FACTOR = 0.1

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM, FEATURE_MAP_G * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(FEATURE_MAP_G * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(FEATURE_MAP_G * 8, FEATURE_MAP_G * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURE_MAP_G * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(FEATURE_MAP_G * 4, FEATURE_MAP_G * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURE_MAP_G * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(FEATURE_MAP_G * 2, FEATURE_MAP_G, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURE_MAP_G),
            nn.ReLU(True),
            nn.ConvTranspose2d(FEATURE_MAP_G, NUM_CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, FEATURE_MAP_D, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(FEATURE_MAP_D, FEATURE_MAP_D * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURE_MAP_D * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(FEATURE_MAP_D * 2, FEATURE_MAP_D * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURE_MAP_D * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(FEATURE_MAP_D * 4, FEATURE_MAP_D * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURE_MAP_D * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(FEATURE_MAP_D * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

netG = Generator().to(device)
netD = Discriminator().to(device)

netG.apply(weights_init)
netD.apply(weights_init)

print("Generator Architecture:")
print(netG)
print("\nDiscriminator Architecture:")
print(netD)

criterion = nn.BCELoss()

fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=device)

real_label = 1.0
fake_label = 0.0

optimizerD = optim.Adam(netD.parameters(), lr=LEARNING_RATE_D, betas=(BETA1, BETA2))
optimizerG = optim.Adam(netG.parameters(), lr=LEARNING_RATE_G, betas=(BETA1, BETA2))

img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
for epoch in range(NUM_EPOCHS):
    for i, data in enumerate(dataloader, 0):

        netD.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)

        if NOISE_FACTOR > 0:
            noise = torch.randn_like(real_cpu) * NOISE_FACTOR
            real_cpu = real_cpu + noise
            real_cpu = torch.clamp(real_cpu, -1, 1)

        label = torch.full((b_size,), real_label - LABEL_SMOOTHING, dtype=torch.float, device=device)

        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label + LABEL_SMOOTHING)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, NUM_EPOCHS, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        if (iters % 500 == 0) or ((epoch == NUM_EPOCHS-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
            plt.figure(figsize=(15, 15))
            plt.subplot(1, 2, 1)
            plt.axis("off")
            plt.title("Real Images")
            plt.imshow(np.transpose(vutils.make_grid(real_cpu[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

            plt.subplot(1, 2, 2)
            plt.axis("off")
            plt.title(f"Generated Images - Epoch {epoch + 1}")
            plt.imshow(np.transpose(vutils.make_grid(fake, padding=2, normalize=True), (1, 2, 0)))
            plt.tight_layout()
            plt.show()

print("Training completed!")

plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

fig = plt.figure(figsize=(8, 8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
plt.show()

with torch.no_grad():
    fake = netG(fixed_noise).detach().cpu()

plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_cpu[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Generated Images")
plt.imshow(np.transpose(vutils.make_grid(fake, padding=2, normalize=True), (1, 2, 0)))
plt.tight_layout()
plt.show()

torch.save(netG.state_dict(), 'generator_final.pth')
torch.save(netD.state_dict(), 'discriminator_final.pth')
print("Models saved successfully!")

print(f"Final Generator Loss: {G_losses[-1]:.4f}")
print(f"Final Discriminator Loss: {D_losses[-1]:.4f}")
