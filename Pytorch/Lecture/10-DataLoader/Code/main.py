import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset

from custom_dataset import CustomDataset
from cnn_model import Model

epochs = 10000

transformations = transforms.Compose([transforms.ToTensor()])
custom_mnist = CustomDataset('../mnist_class/')

mn_dataset_loader = torch.utils.data.DataLoader(dataset=custom_mnist,  batch_size=10, shuffle=False)

model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for epoch in range(epochs):
    l = []
    for i, (images, labels) in enumerate(mn_dataset_loader):
        images = Variable(images)
        labels = Variable(labels)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        l.append(loss.item())
    print(f'Epoch : {epoch} --> Loss: {sum(l)/len(l)}')



