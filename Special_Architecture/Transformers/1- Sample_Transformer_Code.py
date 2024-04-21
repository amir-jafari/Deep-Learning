import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
# -------------------------------------------------------------------------------------------------------

image_size = 224
patch_size = 16
num_classes = 10  # Fashion MNIST has 10 classes
num_channels = 1  # Fashion MNIST images are grayscale
dim = 256
depth = 12
heads = 8
mlp_dim = 512
dropout = 0.1
# -------------------------------------------------------------------------------------------------------
class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, num_channels, dim, depth, heads, mlp_dim, dropout):
        super(VisionTransformer, self).__init__()
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size

        # Image patch embedding
        self.patch_embed = nn.Conv2d(num_channels, dim, kernel_size=patch_size, stride=patch_size)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))

        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout),
            num_layers=depth
        )

        # Classification head
        self.fc = nn.Linear(dim, num_classes)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embed(x)  # (B, C, H, W) -> (B, P, C)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # (B, P, C) -> (B, H, W, C)
        x = x.view(B, -1, C)  # (B, H, W, C) -> (B, P, C)

        # Add positional encoding
        x = torch.cat([x, self.pos_encoding.repeat(B, 1, 1)], dim=1)

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Classification head
        x = self.fc(x)

        return x
# -------------------------------------------------------------------------------------------------------
transform = transforms.Compose([ transforms.ToTensor(), transforms.Resize((224, 224))])

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
# -------------------------------------------------------------------------------------------------------
model = VisionTransformer(image_size, patch_size, num_classes, num_channels, dim, depth, heads, mlp_dim, dropout)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# -------------------------------------------------------------------------------------------------------

num_epochs = 5
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(trainloader)}")
# -------------------------------------------------------------------------------------------------------

# Test the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in tqdm(testloader, desc="Testing"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on the test set: {(100 * correct / total):.2f}%")
