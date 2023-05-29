import torch
#----------------------------------------------------------------------------
dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU
#----------------------------------------------------------------------------
Batch_size = 64     # Batch size
R = 1000            # Input size
S = 100             # Number of neurons
a = 10              # Network output size
#----------------------------------------------------------------------------
p = torch.randn(R, Batch_size).type(dtype)
t = torch.randn(a, Batch_size).type(dtype)
#----------------------------------------------------------------------------
w1 = torch.randn(S, R).type(dtype)
w2 = torch.randn(a, S).type(dtype)

learning_rate = 1e-6
#----------------------------------------------------------------------------
for t in range(500):

    n1 = w1.mm(p)
    a1 = n1.clamp(min=0)
    a2 = w2.mm(a1)



    e = (t - a2).pow(2).sum()
    print(t, e)

    s2 = -2.0 * (t - a2)
    grad_w2 = s2.mm(a1.t())

    s11 = w2.t().mm(s2)
    s1 =s11.clone()
    s1[n1<0] = 0
    grad_w1 = s1.mm(p.t())


    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
# -------------------------------------------------------------------------------------------------------