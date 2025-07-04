import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable


# ========================== Using pretrained model ==========================#
# Download and load pretrained resnet.
resnet = torchvision.models.resnet18(pretrained=True)

# If you want to finetune only the top layer of the model.
for param in resnet.parameters():
    param.requires_grad = False

# Replace top layer for finetuning.
resnet.fc = nn.Linear(resnet.fc.in_features, 100)

# For test.
images = Variable(torch.randn(10, 3, 224, 224))
outputs = resnet(images)
print(outputs.size())  # (10, 100)

# ============================ Save and Load the Model ============================#
# Recommended way: Save and load only the model parameters (state_dict).
torch.save(resnet.state_dict(), 'model_state.pth')  # Save only the state_dict
# To load:
resnet_loaded = torchvision.models.resnet18(pretrained=False)  # Instantiate a new model
resnet_loaded.fc = nn.Linear(resnet_loaded.fc.in_features, 100)  # Recreate the architecture
resnet_loaded.load_state_dict(torch.load('model_state.pth'))  # Load state_dict into the new model
resnet_loaded.eval()  # Set the loaded model to evaluation mode

# Test loaded model
test_outputs = resnet_loaded(images)
print(test_outputs.size())  # (10, 100)