# batch_lm_inference.py
import argparse
import os
import torch
import torch.nn as nn
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--single_mode', type=str, default=None)
args = parser.parse_args()

MODES = [args.single_mode] if args.single_mode else ["char", "sudachi_A", "sudachi_B", "sudachi_C"]


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 100  # max decode steps

class LSTMLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.emb(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

def load_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f]
    stoi = {s: i for i, s in enumerate(vocab)}
    itos = {i: s for s, i in stoi.items()}
    return vocab, stoi, itos

def apply_lm_to_sentence(tokens, model, stoi, itos):
    ids = [stoi.get(t, stoi["<unk>"]) for t in tokens]
    input_tensor = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        logits, _ = model(input_tensor)
    preds = torch.argmax(logits, dim=-1).squeeze(0)
    return [itos[idx.item()] for idx in preds]

# === MAIN LOOP ===
for mode in MODES:
    print(f"Inference using LM for: {mode}")
    input_file = f"/mnt/data/asr_outputs_tokenized/output_{mode}.txt"
    vocab_file = f"/mnt/data/vocab_{mode}.txt"
    checkpoint = f"/mnt/data/lstm_lm_{mode}.pth"
    output_file = f"/mnt/data/asr_lm_outputs_real/output_{mode}.txt"

    vocab, stoi, itos = load_vocab(vocab_file)
    VOCAB_SIZE = len(vocab)

    model = LSTMLM(VOCAB_SIZE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model = model.to(DEVICE).eval()

    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            fname, toks = line.strip().split("\t")
            tokens = toks.strip().split()
            smoothed_tokens = apply_lm_to_sentence(tokens, model, stoi, itos)
            fout.write(f"{fname}\t{' '.join(smoothed_tokens)}\n")

    print(f"Output written to: {output_file}")

