#----------------------------------------------------------------------------
import torch
from torch.autograd import Variable
import numpy as np
#----------------------------------------------------------------------------
Batch_size = 64     # Batch size
R = 1000            # Input size
S = 100             # Number of neurons
a_size = 10              # Network output size
#----------------------------------------------------------------------------
p = Variable(torch.randn(Batch_size, R).cuda())
t = Variable(torch.randn(Batch_size, a_size).cuda(), requires_grad=False)

check1 =p.is_cuda
print("Check inputs and trages are using cuda-------->  " + str(check1) )

model = torch.nn.Sequential(
    torch.nn.Linear(R, S),
    torch.nn.ReLU(),
    torch.nn.Linear(S, a_size),
)

model.cuda()

check2 =next(model.parameters()).is_cuda
print("Check model is using cuda----------------------> " + str(check2) )

performance_index = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
for index in range(500):

    a = model(p)

    loss = performance_index(a, t)
    print(index, loss.item())


    model.zero_grad()

    loss.backward()


    for param in model.parameters():
        param.data -= learning_rate * param.grad.data