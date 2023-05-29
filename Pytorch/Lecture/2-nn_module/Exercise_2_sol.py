#----------------------------------------------------------------------------
import torch
from torch.autograd import Variable

#----------------------------------------------------------------------------
Batch_size = 64     # Batch size
Q = 1000            # Input size
S = 100             # Number of neurons
a_size = 10              # Network output size
#----------------------------------------------------------------------------
torch.manual_seed(1)


p = Variable(torch.randn(Batch_size, Q))
t = Variable(torch.randn(Batch_size, a_size), requires_grad=False)


model = torch.nn.Sequential(
    torch.nn.Linear(Q, S),
    torch.nn.ReLU(),
    torch.nn.Linear(S, a_size),
)
performance_index = torch.nn.MSELoss(reduction='sum')
#----------------------------------------------------------------------------
learning_rate = 1e-4
#----------------------------------------------------------------------------
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate, rho=0.9,eps=1e-6)
optimizer = torch.optim.SGD(model.parameters(),lr= learning_rate, momentum=0.9,weight_decay=0.8)
#----------------------------------------------------------------------------
for index in range(500):

    a = model(p)
    loss = performance_index(a, t)
    print(index, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()