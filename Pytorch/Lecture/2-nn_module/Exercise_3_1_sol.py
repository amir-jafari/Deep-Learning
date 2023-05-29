#----------------------------------------------------------------------------
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt
#----------------------------------------------------------------------------
R = 1            # Input size
S = 20            # Number of neurons
a_size = 1              # Network output size
num_epochs = 50000
#----------------------------------------------------------------------------
inputs1 = np.linspace(-3,3,200, dtype=np.float32).reshape(-1,1)
#targets1 =   np.exp(-np.abs(inputs1),  dtype=np.float32) * np.sin((np.pi*inputs1)).reshape(-1,1)
targets1 =  0.1*np.power(inputs1, 2) * np.sin((inputs1)).reshape(-1,1)


#inputs1 = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
#                    [9.779], [6.182], [7.59], [2.167], [7.042],
#                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)
#
#targets1 = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
#                    [3.366], [2.596], [2.53], [1.221], [2.827],
#                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

p = Variable(torch.from_numpy(inputs1).cuda())
t = Variable(torch.from_numpy(targets1).cuda())

# p = Variable(torch.randn(Batch_size, R).cuda())
# t = Variable(torch.randn(Batch_size, a_size).cuda(), requires_grad=False)

check1 =p.is_cuda
print("Check inputs and trages are using cuda-------->  " + str(check1) )

model = torch.nn.Sequential(
    torch.nn.Linear(R, S),
    torch.nn.Tanh(),
    torch.nn.Linear(S, a_size),
)

model.cuda()

check2 =next(model.parameters()).is_cuda
print("Check model is using cuda---------------------> " + str(check2) )

performance_index = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-1
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    # Convert numpy array to torch Variable
    inputs = p
    targets = t

    # Forward + Backward + Optimize
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print('Epoch [%d/%d], Loss: %.4f' % (epoch + 1, num_epochs, loss.item()))


zz = model(p)

zz1 = zz.data.cpu().numpy()

plt.figure(1)
plt.scatter(inputs1, targets1,c='Red')
plt.hold(True)
plt.scatter(inputs1,zz1)
plt.show()

