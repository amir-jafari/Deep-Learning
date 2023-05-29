#----------------------------------------------------------------------------
import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt
#----------------------------------------------------------------------------
Batch_size = 64     # Batch size
Q = 1000            # Input size
S = 100             # Number of neurons
a_size = 10              # Network output size
#----------------------------------------------------------------------------
S1 = 50
S2 = 20
#----------------------------------------------------------------------------
p = Variable(torch.randn(Batch_size, Q))
t = Variable(torch.randn(Batch_size, a_size), requires_grad=False)
#----------------------------------------------------------------------------

#model = torch.nn.Sequential(
#    torch.nn.Linear(Q, S),
#    torch.nn.ReLU(),
#    torch.nn.Linear(S, a_size),
#)
#----------------------------------------------------------------------------
model = torch.nn.Sequential(
    torch.nn.Linear(Q, S),
    torch.nn.ReLU(),
    torch.nn.Linear(S, S1),
    torch.nn.ReLU(),
    torch.nn.Linear(S1, S2),
    torch.nn.ReLU(),
    torch.nn.Linear(S2, a_size),
)
#----------------------------------------------------------------------------
performance_index = torch.nn.MSELoss(reduction='sum')
#----------------------------------------------------------------------------
learning_rate = 1e-4
epoch = []
performance = []

#----------------------------------------------------------------------------
for key in model.state_dict():
    value = model.state_dict().get(key)
    print(key, value.size())
#----------------------------------------------------------------------------
grad = []
for index in range(500):

    a = model(p)

    loss = performance_index(a, t)
    print(index, loss.item())

    epoch.append(index)
    performance.append(loss.item())



    model.zero_grad()

    loss.backward()

    index = 0
    for param in model.parameters():
        param.data -= learning_rate * param.grad.data

        if index==0:
            w1 = param.grad.data[0]
        index += 1
    grad.append(w1.sum())

plt.figure(1)
plt.loglog(epoch, performance)


plt.figure(2)
plt.plot(grad)
plt.show()

