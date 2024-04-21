import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from transformers import AutoModelForImageClassification, AutoConfig, AdamW

# Define transform to normalize data
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert images to RGB
    transforms.ToTensor(),
    transforms.Resize((224, 224))  # Resize images to match the expected input size of the model
])

# Load Fashion MNIST dataset
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Load pre-trained Vision Transformer model
model_name = "google/vit-base-patch16-224-in21k"
config = AutoConfig.from_pretrained(model_name,
                                    num_labels=10)  # Change num_labels to match the number of classes in Fashion MNIST
model = AutoModelForImageClassification.from_pretrained(model_name, config=config)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=1e-5)

# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(trainloader)}")

# Test the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in tqdm(testloader, desc="Testing"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on the test set: {(100 * correct / total):.2f}%")
