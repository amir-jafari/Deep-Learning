import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm


class PatchEmbedding(nn.Module):
    """Convert image patches to embeddings using convolution"""
    
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.flatten = nn.Flatten(start_dim=2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = x.permute(0, 2, 1)
        return x


class PositionalEmbedding(nn.Module):
    """Add learnable positional embeddings and class token"""
    
    def __init__(self, num_patches=196, embed_dim=768, dropout_rate=0.1):
        super().__init__()
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        batch_size = x.shape[0]
        class_token = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([class_token, x], dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)
        return x


class TransformerEncoder(nn.Module):
    """Single transformer encoder block with multi-head attention and MLP"""
    
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4, dropout_rate=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x):
        # Multi-head attention with residual connection
        residual = x
        x = self.norm1(x)
        x, _ = self.attention(x, x, x)
        x = x + residual
        
        # MLP with residual connection
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        
        return x


class ClassificationHead(nn.Module):
    """Classification head for final predictions"""
    
    def __init__(self, embed_dim=768, num_classes=10):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        x = self.norm(x)
        x = self.head(x)
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) model for image classification"""
    
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=10,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        dropout_rate=0.1
    ):
        super().__init__()
        
        num_patches = (image_size // patch_size) ** 2
        
        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        self.pos_embedding = PositionalEmbedding(
            num_patches=num_patches,
            embed_dim=embed_dim,
            dropout_rate=dropout_rate
        )
        
        self.transformer_encoders = nn.Sequential(*[
            TransformerEncoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout_rate=dropout_rate
            ) for _ in range(depth)
        ])
        
        self.classification_head = ClassificationHead(
            embed_dim=embed_dim,
            num_classes=num_classes
        )
    
    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.pos_embedding(x)
        x = self.transformer_encoders(x)
        x = x[:, 0]  # Use class token for classification
        x = self.classification_head(x)
        return x


def create_data_loaders(batch_size=64):
    """Create training and testing data loaders for CIFAR-10"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return trainloader, testloader


def train_model(model, trainloader, testloader, num_epochs=10, learning_rate=0.001):
    """Train the Vision Transformer model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Training on device: {device}")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(trainloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Evaluate on test set
        if (epoch + 1) % 2 == 0:
            test_accuracy = evaluate_model(model, testloader, device)
            print(f"Test Accuracy: {test_accuracy:.2f}%")


def evaluate_model(model, testloader, device):
    """Evaluate the model on test data"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(testloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def main():
    """Main function to run the Vision Transformer training"""
    # Model configuration
    config = {
        'image_size': 224,
        'patch_size': 16,
        'in_channels': 3,
        'num_classes': 10,
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4,
        'dropout_rate': 0.1
    }
    
    # Create model
    model = VisionTransformer(**config)
    
    # Create data loaders
    trainloader, testloader = create_data_loaders(batch_size=32)
    
    # Train model
    train_model(model, trainloader, testloader, num_epochs=5, learning_rate=0.0001)
    
    # Final evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_accuracy = evaluate_model(model, testloader, device)
    print(f"Final Test Accuracy: {final_accuracy:.2f}%")


if __name__ == "__main__":
    main()