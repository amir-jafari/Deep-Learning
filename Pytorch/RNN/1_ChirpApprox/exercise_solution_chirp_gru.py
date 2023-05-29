# %% -------------------------------------------------------------------------------------------------------------------

# -------------------------------------
# Use a GRU to forecast a Chirp signal
# -------------------------------------

# %% --------------------------------------- Imports -------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import chirp
import matplotlib.pyplot as plt

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
PLOT_SIGNAL, PLOT_RESULT = False, True

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 1e-2
N_EPOCHS = 30
BATCH_SIZE = 128
DROPOUT = 0
SEQ_LEN = 50
HIDDEN_SIZE = 16
N_LAYERS = 1
STATEFUL = True
LOSS_WHOLE_SEQ = False


# %% -------------------------------------- GRU Class ------------------------------------------------------------------
class ChirpGRU(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE, n_layers=N_LAYERS, dropout=DROPOUT):
        super(ChirpGRU, self).__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, p, hidden_state):
        gru_out, h_states = self.gru(p, hidden_state)
        return self.out(gru_out), h_states


# %% -------------------------------------- Data Prep ------------------------------------------------------------------
time_steps = np.linspace(0, 10, 5000)
x = chirp(time_steps, f0=1, f1=0.1, t1=10, method='linear')
if PLOT_SIGNAL:
    plt.plot(time_steps, x)
    plt.show()
x_train, x_test = x[:int(0.75*len(x))], x[int(0.75*len(x)):]
x_train_prep = np.empty((SEQ_LEN, len(x_train)-SEQ_LEN, 1))
for idx in range(len(x_train)-SEQ_LEN):
    x_train_prep[:, idx, :] = x_train[idx:SEQ_LEN+idx].reshape(-1, 1)
x_test_prep = np.empty((SEQ_LEN, len(x_test)-SEQ_LEN, 1))
for idx in range(len(x_test)-SEQ_LEN):
    x_test_prep[:, idx, :] = x_test[idx:SEQ_LEN+idx].reshape(-1, 1)
x_train, x_test = torch.Tensor(x_train_prep).to(device), torch.Tensor(x_test_prep).to(device)
x_train.requires_grad = True
del x_train_prep, x_test_prep

# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = ChirpGRU().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# %% -------------------------------------- Training Loop ----------------------------------------------------------
print("Starting training loop...")
for epoch in range(N_EPOCHS):

    h_state = torch.zeros(N_LAYERS, BATCH_SIZE, HIDDEN_SIZE).float().to(device)
    loss_train = 0
    model.train()
    for batch in range(x_train.shape[1]//BATCH_SIZE):
        inp_inds = slice(batch*BATCH_SIZE, (batch+1)*BATCH_SIZE)
        tar_inds = slice(batch*BATCH_SIZE+1, (batch+1)*BATCH_SIZE+1)
        optimizer.zero_grad()
        pred, h_state = model(x_train[:, inp_inds, :], h_state)
        if STATEFUL:
            h_state = h_state.detach()
        else:
            pass
        if LOSS_WHOLE_SEQ:
            loss = criterion(pred, x_train[:, tar_inds, :])
        else:
            loss = criterion(pred[-1], x_train[-1, tar_inds, :])
        loss.backward()
        optimizer.step()
        loss_train += loss.item()

    h_state = torch.zeros(N_LAYERS, x_test.shape[1]-1, HIDDEN_SIZE).float().to(device)
    model.eval()
    with torch.no_grad():
        pred, h_state = model(x_test[:, :-1, :], h_state)
        loss = criterion(pred, x_test[:, 1:, :])
        loss_test = loss.item()

    print("Epoch {} | Train Loss {:.5f} - Test Loss {:.5f}".format(epoch, loss_train/batch, loss_test))

# %% ------------------------------------------ Final Test -------------------------------------------------------------
if PLOT_RESULT:
    with torch.no_grad():
        h_state = torch.zeros(N_LAYERS, x_train.shape[1], HIDDEN_SIZE).float().to(device)
        pred_train, _ = model(x_train, h_state)
        h_state = torch.zeros(N_LAYERS, x_test.shape[1], HIDDEN_SIZE).float().to(device)
        pred_test, _ = model(x_test, h_state)
    predictions_train, predictions_test = [], []
    for idx in range(pred_train.shape[1]):
        predictions_train.append(pred_train[-1, idx].reshape(-1))
    for idx in range(pred_test.shape[1]):
        predictions_test.append(pred_test[-1, idx].reshape(-1))
    plt.plot(time_steps, x, label="Real Time Series", linewidth=2)
    plt.plot(time_steps[SEQ_LEN:len(predictions_train)+SEQ_LEN],
             np.array(predictions_train), linestyle='dashed', label="Train Prediction")
    plt.scatter(time_steps[len(predictions_train)+2*SEQ_LEN:],
                np.array(predictions_test), color="y", label="Test Prediction")
    plt.title("Chirp function with 1 Hz frequency at t=0 and 0.1 Hz freq at t=10\n"
              "LSTM predictions using the previous {} time steps".format(SEQ_LEN))
    plt.xlabel("Time"); plt.ylabel("Signal Intensity")
    plt.legend()
    plt.show()
