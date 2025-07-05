import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import re

# Dummy data for demonstration
SRC_SENTENCES = ['I am a student.', 'You are a teacher.', 'They are engineers.']
TGT_SENTENCES = ['Je suis un étudiant.', 'Vous êtes un professeur.', 'Ils sont des ingénieurs.']


def tokenize(sentence, lang, word2idx):
    # Add spaces around punctuation
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)  # Remove multiple spaces

    # Split into words and convert to index
    tokens = []
    unk_idx = 3 if '<unk>' in word2idx else None
    for word in sentence.strip().split():
        tokens.append(word2idx.get(word, unk_idx))
    return tokens


def build_vocab(sentences, lang):
    vocab = {}
    for sentence in sentences:
        for word in tokenize(sentence, lang, vocab):
            vocab[word] = vocab.get(word, len(vocab))

    # Add special tokens
    vocab['<sos>'] = len(vocab)
    vocab['<eos>'] = len(vocab)
    vocab['<pad>'] = len(vocab)
    vocab['<unk>'] = len(vocab)

    word2idx = vocab
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word

# Build vocabularies
src_word2idx, src_idx2word = build_vocab(SRC_SENTENCES, 'en')
tgt_word2idx, tgt_idx2word = build_vocab(TGT_SENTENCES, 'fr')


# ... (rest of the code remains the same)


# ... (rest of the code remains the same)
# Encoder
class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.rnn(embedded, hidden)
        return output, hidden


# Decoder
class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.rnn(output, hidden)
        output = self.out(output[0])
        return output, hidden


# Seq2Seq model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[0]
        target_len = target.shape[1]
        target_vocab_size = len(tgt_idx2word)

        outputs = torch.zeros(batch_size, target_len, target_vocab_size)

        # Encoder
        encoder_hidden = torch.zeros(1, batch_size, self.encoder.rnn.hidden_size)
        for i in range(source.shape[1]):
            encoder_output, encoder_hidden = self.encoder(source[:, i], encoder_hidden)

        # Decoder
        decoder_input = torch.tensor([[tgt_word2idx['<sos>']]] * batch_size)
        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < teacher_force_ratio else False

        if use_teacher_forcing:
            for t in range(target_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs[:, t] = decoder_output
                decoder_input = target[:, t].unsqueeze(1)
        else:
            for t in range(target_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs[:, t] = decoder_output
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()

        return outputs


# Instantiate models
embed_size = 256
hidden_size = 512
encoder = Encoder(len(src_idx2word), embed_size, hidden_size)
decoder = Decoder(len(tgt_idx2word), embed_size, hidden_size)
model = Seq2Seq(encoder, decoder)

# Training loop
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    for idx in range(len(SRC_SENTENCES)):
        src_tokens = tokenize(SRC_SENTENCES[idx], 'en', src_word2idx) + [src_word2idx['<eos>']]
        tgt_tokens = tokenize(TGT_SENTENCES[idx], 'fr', tgt_word2idx) + [tgt_word2idx['<eos>']]
        src_tensor = torch.tensor([src_tokens])
        tgt_tensor = torch.tensor([tgt_tokens])

        optimizer.zero_grad()
        outputs = model(src_tensor, tgt_tensor)
        loss = criterion(outputs.view(-1, len(tgt_idx2word)), tgt_tensor.view(-1))
        loss.backward()
        optimizer.step()

# Inference
source = tokenize('I am a student.', 'en', src_word2idx) + [src_word2idx['<eos>']]
source_tensor = torch.tensor([source])
max_len = 20  # Maximum length of the target sentence

encoder_hidden = torch.zeros(1, 1, hidden_size)
for i in range(len(source)):
    encoder_output, encoder_hidden = encoder(source_tensor[:, i], encoder_hidden)

decoder_input = torch.tensor([[tgt_word2idx['<sos>']]])
decoder_hidden = encoder_hidden

translated = []
for t in range(max_len):
    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
    _, topi = decoder_output.topk(1)
    decoded = topi.item()
    if decoded == tgt_word2idx['<eos>']:
        break
    else:
        translated.append(tgt_idx2word[decoded])
        decoder_input = topi.squeeze().detach()

print(' '.join(translated))