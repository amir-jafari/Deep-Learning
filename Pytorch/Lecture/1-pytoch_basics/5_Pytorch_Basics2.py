import torchvision

import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
import torchvision.utils as vutils

# Define CIFAR-10 class names
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Define transformation for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images
])

# Download and construct dataset
train_dataset = dsets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = dsets.CIFAR10(root='./data', train=False, transform=transform, download=True)

# Data Loader (provides queue and thread in a very simple way)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False, num_workers=2)

# Iterate through the DataLoader
data_iter = iter(train_loader)
images, labels = next(data_iter)

# Print shape of images and labels
print("Image batch dimensions:", images.size())
print("Label batch dimensions:", labels.size())

# Plot a few images from the dataset
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Show a batch of images
imshow(vutils.make_grid(images))
print(' '.join(f'{cifar10_classes[labels[j]]:5s}' for j in range(4)))
