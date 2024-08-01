# =================================Import Library=======================================================================
# pip install pykan
# https://kindxiaoming.github.io/pykan/index.html


import torch
import torch.nn as nn
import torch.optim as optim


class KAN(nn.Module):
    def __init__(self, width, grid, k):
        super(KAN, self).__init__()
        self.width = width
        self.grid = grid
        self.k = k

        # Define the network layers
        layers = []
        input_dim = width[0]
        for w in width[1:]:
            layers.append(nn.Linear(input_dim, w))
            layers.append(nn.ReLU())
            input_dim = w
        layers.append(nn.Linear(input_dim, 1))  # Output layer
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def train_model(self, dataset, optimizer='Adam', steps=100):
        self.train()  # Set the model to training mode
        # Split dataset into inputs and targets
        inputs, targets = dataset['inputs'], dataset['targets']

        if optimizer == 'Adam':
            optimizer = optim.Adam(self.parameters())
        elif optimizer == 'SGD':
            optimizer = optim.SGD(self.parameters(), lr=0.01)
        else:
            raise ValueError("Unsupported optimizer type")

        criterion = nn.MSELoss()  # Example loss function

        for step in range(steps):
            optimizer.zero_grad()
            outputs = self(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print(f"Step {step}/{steps}, Loss: {loss.item()}")

    def save_ckpt(self, path):
        torch.save(self.state_dict(), path)

    def load_ckpt(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()  # Set the model to evaluation mode

    def plot(self):
        # Example plotting function (depends on the specific KAN implementation)
        pass


# Example dataset creation function
def create_dataset(f, n_var):
    # Generate a dummy dataset based on function f
    inputs = torch.rand((100, n_var))  # 100 samples, n_var features
    targets = f(inputs)
    return {'inputs': inputs, 'targets': targets}


# Usage example
model = KAN(width=[2, 5, 3, 1], grid=3, k=3)

# Create a dataset
f = lambda x: torch.sin(x[:, [0]]) + x[:, [1]] ** 2
dataset = create_dataset(f, n_var=2)

# Train the model
model.train_model(dataset, optimizer='Adam', steps=20)

# Save the model checkpoint
model.save_ckpt('model.ckpt')

# Load the model
model2 = KAN(width=[2, 5, 3, 1], grid=3, k=3)
model2.load_ckpt('model.ckpt')

# Make predictions
inputs = dataset['inputs']
predictions = model2(inputs).detach().numpy()
print(predictions)
