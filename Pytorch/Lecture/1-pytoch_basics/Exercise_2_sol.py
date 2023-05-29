import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
#----------------------------------------------------------------------------
dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU
#----------------------------------------------------------------------------
Batch_size = 64     # Batch size
Q = 1000            # Input size
S = 100             # Number of neurons
a = 10              # Network output size
#----------------------------------------------------------------------------

p = Variable(torch.randn(Batch_size, Q).type(dtype), requires_grad=False)
tar = Variable(torch.randn(Batch_size, a).type(dtype), requires_grad=False)

w1 = Variable(torch.randn(Q, S).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(S, a).type(dtype), requires_grad=True)


learning_rate = 1e-6
#----------------------------------------------------------------------------
epoch = []
performance = []
weighT_Grad1 = []
weighT_Grad2 = []
for t in range(500):

    a_net = p.mm(w1).clamp(min=0).mm(w2)
    loss = (a_net - tar).pow(2).sum()
    print(t, loss.item())

    epoch.append(t)
    performance.append(loss.item())

    loss.backward()

    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data

    weighT_Grad1.append(w1.grad.data.sum())
    weighT_Grad2.append(w2.grad.data.sum())


    w1.grad.data.zero_()
    w2.grad.data.zero_()


plt.figure(1)
plt.plot(epoch, performance)

plt.figure(2)
plt.semilogy(epoch, performance)

plt.figure(3)
plt.plot(epoch, weighT_Grad1)

plt.figure(4)
plt.plot(epoch, weighT_Grad2)



plt.show()