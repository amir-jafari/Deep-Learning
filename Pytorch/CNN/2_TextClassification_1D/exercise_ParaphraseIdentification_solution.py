# %% -------------------------------------------------------------------------------------------------------------------

# % ----------------------------------------------------------
# Train a 1D-CNN to tell whether two sentences are paraphrases
# % ----------------------------------------------------------

# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
import numpy as np
import json
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import nltk
nltk.download('punkt')

if "msr_paraphrase_train.txt" not in os.listdir(os.getcwd()) or "msr_paraphrase_test.txt" not in os.listdir(os.getcwd()):
    try:
        os.system("wget https://raw.githubusercontent.com/wasiahmad/paraphrase_identification/master/dataset/msr-paraphrase-corpus/msr_paraphrase_train.txt")
        os.system("wget https://raw.githubusercontent.com/wasiahmad/paraphrase_identification/master/dataset/msr-paraphrase-corpus/msr_paraphrase_test.txt")
    except:
        print("There was a problem with the download!")
    if "msr_paraphrase_train.txt" not in os.listdir(os.getcwd()) or "msr_paraphrase_test.txt" not in os.listdir(os.getcwd()):
        print("There was a problem with the download!")
        import sys
        sys.exit()
        # 1. Download the Microsoft Research Paraphrase Corpus from https://gluebenchmark.com/tasks. It's easier to get it from
        # go to https://github.com/wasiahmad/paraphrase_identification/tree/master/dataset/msr-paraphrase-corpus (the .txt files)

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
class Args:
    def __init__(self):
        self.seq_len = "get_max_from_data"
        self.embedding_dim = 50
        self.n_epochs = 30
        self.lr = 1e-3
        self.batch_size = 512
        self.freeze_embeddings = False
        self.train = True
        self.save_model = True

args = Args()

# %% ----------------------------------- Helper Functions --------------------------------------------------------------
# 2. Replace the acc function with an equivalent that returns the f1 score and use this metric for early stopping.
def f1(x, y, return_labels=False):
    with torch.no_grad():
        logits = model(x)
        pred_labels = np.argmax(logits.cpu().numpy(), axis=1)
    if return_labels:
        return pred_labels
    else:
        return 100*f1_score(y.cpu().numpy(), pred_labels)

def extract_vocab_dict_and_msl(sentences_train, sentences_dev):
    """ Tokenizes all the sentences and gets a dictionary of unique tokens and also the maximum sequence length """
    tokens, ms_len = [], 0
    for sentence in list(sentences_train) + list(sentences_dev):
        tokens_in_sentence = nltk.word_tokenize(sentence)
        if ms_len < len(tokens_in_sentence):
            ms_len = len(tokens_in_sentence)
        tokens += tokens_in_sentence
    token_vocab = {key: i for key, i in zip(set(tokens), range(1, len(set(tokens))+1))}
    if len(np.unique(list(token_vocab.values()))) != len(token_vocab):
        "There are some rep words..."
    return token_vocab, ms_len

def convert_to_ids(raw_sentences, vocab_dict, pad_to):
    """ Takes an NumPy array of raw text sentences and converts to a sequence of token ids """
    x = np.empty((len(raw_sentences), pad_to))
    for idx, sentence in enumerate(raw_sentences):
        word_ids = []
        for token in nltk.word_tokenize(sentence):
            try:
                word_ids.append(vocab_dict[token])
            except:
                word_ids.append(vocab_dict[token])
        if pad_to < len(word_ids):
            x[idx] = word_ids[:pad_to]
        else:
            x[idx] = word_ids + [0] * (pad_to - len(word_ids))
    return x

def get_glove_embeddings(vocab_dict):
    with open("glove.6B.50d.txt", "r") as s:
        glove = s.read()
    embeddings_dict = {}
    for line in glove.split("\n")[:-1]:
        text = line.split()
        if text[0] in vocab_dict:
            embeddings_dict[vocab_dict[text[0]]] = torch.from_numpy(np.array(text[1:], dtype="float32"))
    return embeddings_dict

def get_glove_table(vocab_dict, glove_dict):
    lookup_table = torch.empty((len(vocab_dict)+2, 50))
    for token_id in sorted(vocab_dict.values()):
        if token_id in glove_dict:
            lookup_table[token_id] = glove_dict[token_id]
        else:
            lookup_table[token_id] = torch.zeros((1, 50))  # For unknown tokens
    lookup_table[0] = torch.zeros((1, 50))
    return lookup_table

# %% -------------------------------------- CNN Class ------------------------------------------------------------------
# 4. Modify the CNN class so that it works with sentences of 80 words (this should be the maximum sequence length
# extracted from the corpus).
class CNN(nn.Module):
    def __init__(self, vocab_size):
        super(CNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size + 2, args.embedding_dim)

        self.conv1 = nn.Conv1d(args.embedding_dim, args.embedding_dim, 9)
        self.convnorm1 = nn.BatchNorm1d(args.embedding_dim)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(args.embedding_dim, args.embedding_dim, 9)
        self.convnorm2 = nn.BatchNorm1d(args.embedding_dim)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(args.embedding_dim, args.embedding_dim, 9)
        self.conv4 = nn.Conv1d(args.embedding_dim, args.embedding_dim, 6)
        self.linear = nn.Linear(args.embedding_dim, 2)

        self.act = torch.relu

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = self.pool1(self.convnorm1(self.act(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.act(self.conv2(x))))
        return self.linear(self.act(self.conv4(self.act(self.conv3(x)))).reshape(-1, args.embedding_dim))


# %% -------------------------------------- Data Prep ------------------------------------------------------------------
# 3. Write a function to load the data from the .txt files. The usual way of pre-processing is stacking both sentences
# together and separating them by a "SEP" special token. The two sentences constitute one input.
def prep_data(path):
    with open(path, 'r', encoding='utf8') as s:
        a = s.read()
    x, y = [], []
    for line in a.split("\n")[1:]:
        line_split = line.split("\t")
        try:
            x.append(line_split[3] + " SEP " + line_split[4])  # We make one input with the two sentences
            y.append(int(line_split[0]))  # Separating them with the special token "SEP"
        except Exception as e:
            print("There was an error loading this line:", e)
            print(line)
            print(line_split)
    return np.array(x), np.array(y)

x_train_raw, y_train = prep_data("msr_paraphrase_train.txt")
x_dev_raw, y_dev = prep_data("msr_paraphrase_test.txt")

try:
    with open("exercise_prep_data/vocab_dict.json", "r") as s:
        token_ids = json.load(s)
    msl = np.load("exercise_prep_data/max_sequence_length.npy").item()
except:
    print("Tokenizing all the examples to get a vocab dict and the maximum sequence length...")
    token_ids, msl = extract_vocab_dict_and_msl(x_train_raw, x_dev_raw)
    os.mkdir("exercise_prep_data")
    with open("exercise_prep_data/vocab_dict.json", "w") as s:
        json.dump(token_ids, s)
    np.save("exercise_prep_data/max_sequence_length.npy", np.array([msl]))
if args.seq_len == "get_max_from_data":
    args.seq_len = msl

glove_embeddings = get_glove_embeddings(token_ids)

try:
    x_train = np.load("exercise_prep_data/prep_train_len{}.npy".format(args.seq_len))
    x_dev = np.load("exercise_prep_data/prep_dev_len{}.npy".format(args.seq_len))
except:
    print("Converting all the sentences to sequences of token ids...")
    x_train = convert_to_ids(x_train_raw, token_ids, args.seq_len)
    np.save("exercise_prep_data/prep_train_len{}.npy".format(args.seq_len), x_train)
    x_dev = convert_to_ids(x_dev_raw, token_ids, args.seq_len)
    np.save("exercise_prep_data/prep_dev_len{}.npy".format(args.seq_len), x_dev)
del x_train_raw, x_dev_raw

x_train, x_dev = torch.LongTensor(x_train).to(device), torch.LongTensor(x_dev).to(device)
y_train, y_dev = torch.LongTensor(y_train).to(device), torch.LongTensor(y_dev).to(device)

# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = CNN(len(token_ids)).to(device)
look_up_table = get_glove_table(token_ids, glove_embeddings)
model.embedding.weight.data.copy_(look_up_table)
if args.freeze_embeddings:
    model.embedding.weight.requires_grad = False
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()

# %% -------------------------------------- Training Loop ----------------------------------------------------------
labels_ditrib = torch.unique(y_dev, return_counts=True)
print("The no information rate is {:.2f}".format(100*labels_ditrib[1].max().item()/len(y_dev)))
if args.train:
    f1_dev_best = 0
    print("Starting training loop...")
    for epoch in range(args.n_epochs):

        loss_train = 0
        model.train()
        for batch in range(len(x_train)//args.batch_size + 1):
            inds = slice(batch*args.batch_size, (batch+1)*args.batch_size)
            optimizer.zero_grad()
            logits = model(x_train[inds])
            loss = criterion(logits, y_train[inds])
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        model.eval()
        with torch.no_grad():
            y_dev_pred = model(x_dev)
            loss = criterion(y_dev_pred, y_dev)
            loss_test = loss.item()

        f1_dev = f1(x_dev, y_dev)
        print("Epoch {} | Train Loss {:.5f}, Train f1 {:.2f} - Test Loss {:.5f}, Test f1 {:.2f}".format(
            epoch, loss_train/(len(x_train)//args.batch_size), f1(x_train, y_train), loss_test, f1_dev))

        if f1_dev > f1_dev_best and args.save_model:
            torch.save(model.state_dict(), "cnn_pharaprase_identifcation.pt")
            print("The model has been saved!")
            f1_dev_best = f1_dev

# %% ------------------------------------------ Final test -------------------------------------------------------------
model.load_state_dict(torch.load("cnn_pharaprase_identifcation.pt"))
model.eval()
y_test_pred = f1(x_dev, y_dev, return_labels=True)
print("The accuracy on the test set is {:.2f}".format(100*accuracy_score(y_dev.cpu().numpy(), y_test_pred), "%"))
print("The confusion matrix is")
print(confusion_matrix(y_dev.cpu().numpy(), y_test_pred))
print("The f1-score is:", f1_score(y_dev.cpu().numpy(), y_test_pred))
