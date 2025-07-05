import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import math

# ============================================================================
# SETUP AND CONFIGURATION
# ============================================================================

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

def change_image_shape(images):
    if len(images.shape) == 3:
        images = images.unsqueeze(1)
    return images

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

x_train = train_dataset.data.float() / 255.0
x_test = test_dataset.data.float() / 255.0
y_train = train_dataset.targets
y_test = test_dataset.targets

x_train = change_image_shape(x_train)
x_test = change_image_shape(x_test)
img_size = x_train[0].shape
n_classes = len(torch.unique(y_train))

T = 1000
beta_start = 0.0001
beta_end = 0.02

# ============================================================================
# DIFFUSION UTILITIES
# ============================================================================

def get_beta_schedule(T, beta_start=0.0001, beta_end=0.02):
    return np.linspace(beta_start, beta_end, T)

def get_alpha_schedule(betas):
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    return alphas, alphas_cumprod

betas = get_beta_schedule(T, beta_start, beta_end)
alphas, alphas_cumprod = get_alpha_schedule(betas)

def forward_diffusion(x0, t, alphas_cumprod):
    batch_size = x0.shape[0]
    device = x0.device

    alpha_cumprod_t = torch.tensor(alphas_cumprod)[t].to(device)
    alpha_cumprod_t = alpha_cumprod_t.view(batch_size, 1, 1, 1)

    noise = torch.randn_like(x0)

    sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
    sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)

    noisy_image = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise

    return noisy_image, noise

def sinusoidal_embedding(timesteps, embedding_dim):
    device = timesteps.device
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb

# ============================================================================
# NEURAL NETWORK COMPONENTS
# ============================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm(channels)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        x_reshaped = x.view(batch_size, channels, height * width).permute(0, 2, 1)
        attn_output, _ = self.mha(x_reshaped, x_reshaped, x_reshaped)
        x_reshaped = x_reshaped + attn_output
        x_reshaped = self.ln(x_reshaped)
        return x_reshaped.permute(0, 2, 1).view(batch_size, channels, height, width)

class UNet(nn.Module):
    def __init__(self, img_channels=1, time_embedding_dim=128):
        super().__init__()
        self.time_embedding_dim = time_embedding_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.ReLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.ReLU()
        )

        self.conv1 = ConvBlock(img_channels, 64)
        self.res1 = ResidualBlock(64)

        self.conv2 = ConvBlock(64, 128, stride=2)
        self.res2 = ResidualBlock(128)

        self.conv3 = ConvBlock(128, 256, stride=2)
        self.res3 = ResidualBlock(256)

        self.conv4 = ConvBlock(256, 512, stride=2)
        self.res4 = ResidualBlock(512)
        self.attn = AttentionBlock(512)

        self.time_proj = nn.Linear(time_embedding_dim, 512)

        self.upconv1 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
        self.upbn1 = nn.BatchNorm2d(256)
        self.upres1 = ResidualBlock(512)

        self.upconv2 = nn.ConvTranspose2d(512, 128, 3, stride=2, padding=1, output_padding=1)
        self.upbn2 = nn.BatchNorm2d(128)
        self.upres2 = ResidualBlock(256)

        self.upconv3 = nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1)
        self.upbn3 = nn.BatchNorm2d(64)
        self.upres3 = ResidualBlock(128)

        self.output_conv = nn.Conv2d(128, img_channels, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        time_emb = sinusoidal_embedding(t, self.time_embedding_dim)
        time_emb = self.time_mlp(time_emb)

        x1 = self.res1(self.conv1(x))
        x2 = self.res2(self.conv2(x1))
        x3 = self.res3(self.conv3(x2))
        x4 = self.res4(self.conv4(x3))
        x4 = self.attn(x4)

        time_emb_proj = self.time_proj(time_emb).view(-1, 512, 1, 1)
        x4 = x4 + time_emb_proj

        x = self.relu(self.upbn1(self.upconv1(x4)))
        x = F.interpolate(x, size=x3.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x3], dim=1)
        x = self.upres1(x)

        x = self.relu(self.upbn2(self.upconv2(x)))
        x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x2], dim=1)
        x = self.upres2(x)

        x = self.relu(self.upbn3(self.upconv3(x)))
        x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x1], dim=1)
        x = self.upres3(x)

        return self.output_conv(x)

# ============================================================================
# DIFFUSION MODEL CLASS
# ============================================================================

class DiffusionModel:
    def __init__(self, img_channels=1, T=1000):
        self.img_channels = img_channels
        self.T = T
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UNet(img_channels).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

        self.betas = torch.tensor(get_beta_schedule(T), dtype=torch.float32).to(self.device)
        self.alphas = torch.tensor(alphas, dtype=torch.float32).to(self.device)
        self.alphas_cumprod = torch.tensor(alphas_cumprod, dtype=torch.float32).to(self.device)

        self.losses = []

    def train_step(self, x_batch):
        batch_size = x_batch.shape[0]
        x_batch = x_batch.to(self.device)

        t = torch.randint(0, self.T, (batch_size,), device=self.device)

        noisy_images, noise = forward_diffusion(x_batch, t, self.alphas_cumprod)
        predicted_noise = self.model(noisy_images, t)
        loss = F.mse_loss(noise, predicted_noise)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, x_train, epochs=100, batch_size=32):
        dataset = TensorDataset(x_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            epoch_losses = []

            for batch in dataloader:
                loss = self.train_step(batch[0])
                epoch_losses.append(loss)

            avg_loss = np.mean(epoch_losses)
            self.losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}')
                self.sample_images(num_samples=8)

    def sample_images(self, num_samples=8):
        self.model.eval()
        with torch.no_grad():
            x = torch.randn(num_samples, self.img_channels, 28, 28).to(self.device)

            for t in reversed(range(0, self.T, 50)):
                t_batch = torch.full((num_samples,), t, device=self.device)

                predicted_noise = self.model(x, t_batch)

                alpha_t = self.alphas[t]
                alpha_cumprod_t = self.alphas_cumprod[t]
                beta_t = self.betas[t]

                x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise)

                if t > 0:
                    noise = torch.randn_like(x)
                    x = x + torch.sqrt(beta_t) * noise

            x = torch.clamp(x, 0.0, 1.0)
            self.plot_images(x.cpu().numpy())
        self.model.train()

    def plot_images(self, images):
        n = int(np.sqrt(len(images)))
        plt.figure(figsize=(n * 2, n * 2))

        for i in range(len(images)):
            plt.subplot(n, n, i + 1)
            plt.imshow(images[i].squeeze(), cmap='gray')
            plt.axis('off')

        plt.tight_layout()
        plt.show()

# ============================================================================
# TRAINING AND EXECUTION
# ============================================================================

print("Creating Diffusion Model...")
diffusion_model = DiffusionModel(img_channels=1, T=T)

print("Starting training...")
diffusion_model.train(x_train, epochs=50, batch_size=1500)

plt.figure(figsize=(10, 5))
plt.plot(diffusion_model.losses, label='Diffusion Loss', color='blue', linewidth=2)
plt.title('Diffusion Model Training Loss', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

print("Generating final samples...")
diffusion_model.sample_images(num_samples=16)
