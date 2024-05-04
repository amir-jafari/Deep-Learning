# =================================Import Library=======================================================================
# pip install pykan
# https://kindxiaoming.github.io/pykan/index.html
from kan import *
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
model = KAN(width=[2,3,1], grid=3, k=3)

# create dataset
f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
dataset = create_dataset(f, n_var=2)
# ----------------------------------------------------------------------------------------------------------------------
model.train(dataset, opt="LBFGS", steps=20)
# ----------------------------------------------------------------------------------------------------------------------
model.save_ckpt('model.ckpt')
model.plot()
plt.show()
print(model)
# ----------------------------------------------------------------------------------------------------------------------
model2 = KAN(width=[2,1,1], grid=3, k=3)
model2.load_ckpt('model.ckpt')
pred = model.forward(dataset['train_input']).detach().numpy()