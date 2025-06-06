# train_lstm_lm.py
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from pathlib import Path
import argparse

# ==== Configs ====

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True)
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--vocab', type=str, required=True)
parser.add_argument('--checkpoint', type=str, required=True)
args = parser.parse_args()

MODE = args.mode
DATA_PATH = args.data
VOCAB_PATH = args.vocab
CHECKPOINT_PATH = args.checkpoint

EMBED_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
BATCH_SIZE = 32
EPOCHS = 10
SEQ_LEN = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==== Vocabulary ====
with open(DATA_PATH, "r", encoding="utf-8") as f:
    tokens = f.read().split()
vocab_counter = Counter(tokens)
vocab = ["<pad>", "<unk>"] + sorted(vocab_counter)
stoi = {s: i for i, s in enumerate(vocab)}
itos = {i: s for s, i in stoi.items()}

with open(VOCAB_PATH, "w", encoding="utf-8") as f:
    for token in vocab:
        f.write(token + "\n")

VOCAB_SIZE = len(vocab)

# ==== Dataset ====
class LMDataset(Dataset):
    def __init__(self, token_ids, seq_len):
        self.data = token_ids
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, i):
        return (
            torch.tensor(self.data[i:i+self.seq_len], dtype=torch.long),
            torch.tensor(self.data[i+1:i+self.seq_len+1], dtype=torch.long),
        )

token_ids = [stoi.get(t, stoi["<unk>"]) for t in tokens]
dataset = LMDataset(token_ids, SEQ_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==== Model ====
class LSTMLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.emb(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

model = LSTMLM(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ==== Training ====
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits, _ = model(xb)
        loss = criterion(logits.view(-1, VOCAB_SIZE), yb.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

# ==== Save ====
torch.save(model.state_dict(), CHECKPOINT_PATH)
print(f"Saved model to {CHECKPOINT_PATH}")

