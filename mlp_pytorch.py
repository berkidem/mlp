"""
Implements a simple n-gram language model in PyTorch.
Acts as the correctness reference for all the other versions.
"""
import math
import pickle
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import datetime
import csv

from common import RNG, StepTimer

# -----------------------------------------------------------------------------
# PyTorch implementation of the MLP n-gram model: first without using nn.Module

class MLPRaw:
    """
    Takes the previous n tokens, encodes them with a lookup table,
    concatenates the vectors and predicts the next token with an MLP.

    Reference:
    Bengio et al. 2003 https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
    """

    def __init__(self, vocab_size, context_length, embedding_size, hidden_size, rng):
        v, t, e, h = vocab_size, context_length, embedding_size, hidden_size
        self.embedding_size = embedding_size
        self.wte = torch.tensor(rng.randn(v * e, mu=0, sigma=1.0)).view(v, e)
        scale = 1 / math.sqrt(e * t)
        self.fc1_weights =  torch.tensor(rng.rand(t * e * h, -scale, scale)).view(h, t * e).T
        self.fc1_bias = torch.tensor(rng.rand(h, -scale, scale))
        scale = 1 / math.sqrt(h)
        self.fc2_weights = torch.tensor(rng.rand(v * h, -scale, scale)).view(v, h).T
        self.fc2_bias = torch.tensor(rng.rand(v, -scale, scale))
        # Have to explicitly tell PyTorch that these are parameters and require gradients
        for p in self.parameters():
            p.requires_grad = True

    def parameters(self):
        return [self.wte, self.fc1_weights, self.fc1_bias, self.fc2_weights, self.fc2_bias]

    def __call__(self, idx, targets=None):
        return self.forward(idx, targets)

    def forward(self, idx, targets=None):
        # idx are the input tokens, (B, T) tensor of integers
        # targets are the target tokens, (B, ) tensor of integers
        B, T = idx.size()
        # forward pass
        # encode all the tokens using the embedding table
        # FIX: turn off the next row
        idx = idx.long()
        emb = self.wte[idx] # (B, T, embedding_size)
        # concat all of the embeddings together
        emb = emb.view(B, -1) # (B, T * embedding_size)
        # forward through the MLP
        hidden = torch.tanh(emb @ self.fc1_weights + self.fc1_bias)
        logits = hidden @ self.fc2_weights + self.fc2_bias
        # if we are given desired targets, also calculate the loss
        loss = None
        if targets is not None:
            # FIX: turn off the next row
            targets = targets.long()
            loss = F.cross_entropy(logits, targets)
        return logits, loss

# -----------------------------------------------------------------------------
# Equivalent PyTorch implementation of the MLP n-gram model: using nn.Module

class MLP(nn.Module):

    def __init__(self, vocab_size, context_length, embedding_size, hidden_size, rng):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, embedding_size) # token embedding table
        self.mlp = nn.Sequential(
            nn.Linear(context_length * embedding_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, vocab_size)
        )
        self.reinit(rng)

    @torch.no_grad()
    def reinit(self, rng):
        # This function is a bit of a hack and would not be present in
        # typical PyTorch code. Basically:
        # - we want to use our own RNG to initialize the weights.
        # - but we don't want to change idiomatic PyTorch code (above).
        # So here in this function we overwrite the weights using our own RNG.
        # This ensures that we have full control over the initialization and
        # can easily compare the results with other implementations.

        def reinit_tensor_randn(w, mu, sigma):
            winit = torch.tensor(rng.randn(w.numel(), mu=mu, sigma=sigma))
            w.copy_(winit.view_as(w))

        def reinit_tensor_rand(w, a, b):
            winit = torch.tensor(rng.rand(w.numel(), a=a, b=b))
            w.copy_(winit.view_as(w))

        # Let's match the PyTorch default initialization:
        # Embedding with N(0,1)
        reinit_tensor_randn(self.wte.weight, mu=0, sigma=1.0)
        # Linear (both W,b) with U(-K, K) where K = 1/sqrt(fan_in)
        scale = (self.mlp[0].in_features)**-0.5
        reinit_tensor_rand(self.mlp[0].weight, -scale, scale)
        reinit_tensor_rand(self.mlp[0].bias, -scale, scale)
        scale = (self.mlp[2].in_features)**-0.5
        reinit_tensor_rand(self.mlp[2].weight, -scale, scale)
        reinit_tensor_rand(self.mlp[2].bias, -scale, scale)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        emb = self.wte(idx) # (B, T, embedding_size)
        emb = emb.view(B, -1) # (B, T * embedding_size)
        logits = self.mlp(emb)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        return logits, loss

# -----------------------------------------------------------------------------
# simple DataLoader that iterates over all the n-grams

def dataloader(tokens, context_length, batch_size, device=torch.device('cpu')):
    # returns inputs, targets as torch Tensors of shape (B, T), (B, )
    n = len(tokens)
    inputs, targets = [], []
    pos = 0
    while True:
        # simple sliding window over the tokens, of size context_length + 1
        window = tokens[pos:pos + context_length + 1]
        inputs.append(window[:-1])
        targets.append(window[-1])
        # once we've collected a batch, emit it
        if len(inputs) == batch_size:
            # Convert to long tensors (int64) before sending to device
            yield (torch.tensor(inputs, dtype=torch.long).to(device), 
                   torch.tensor(targets, dtype=torch.long).to(device))
            inputs, targets = [], []
        # advance the position and wrap around if we reach the end
        pos += 1
        if pos + context_length >= n:
            pos = 0

# -----------------------------------------------------------------------------
# evaluation function

@torch.inference_mode()
def eval_split(model, tokens, max_batches=None, device=torch.device('cpu')):
    # calculate the loss on the given tokens
    total_loss = 0
    num_batches = len(tokens) // batch_size
    if max_batches is not None:
        num_batches = min(num_batches, max_batches)
    data_iter = dataloader(tokens, context_length, batch_size, device)
    for _ in range(num_batches):
        inputs, targets = next(data_iter)
        logits, loss = model(inputs, targets)
        total_loss += loss.item()
    mean_loss = total_loss / num_batches
    return mean_loss

# -----------------------------------------------------------------------------
# sampling from the model

def softmax(logits):
    # logits here is a (1D) torch.Tensor of shape (V,)
    maxval = torch.max(logits) # subtract max for numerical stability
    exps = torch.exp(logits - maxval)
    probs = exps / torch.sum(exps)
    return probs

def sample_discrete(probs, coinf):
    # sample from a discrete distribution
    # probs is a torch.Tensor of shape (V,)
    cdf = 0.0
    for i, prob in enumerate(probs):
        cdf += prob
        if coinf < cdf:
            return i
    return len(probs) - 1  # in case of rounding errors

# -----------------------------------------------------------------------------
# let's train!

# "train" the Tokenizer, so we're able to map between characters and tokens
with open("data/train_sequences.pkl", "rb") as f:
    train_text = pickle.load(f)

max_num = max(max(seq) for seq in train_text)
min_num = min(min(seq) for seq in train_text)
assert (min_num == 0) & (max_num == 1016)
uchars = list(set([item for sublist in train_text for item in sublist]))
uchars = sorted(uchars)
vocab_size = len(uchars)
print(f"vocab_size: {vocab_size}")

encoding = pd.read_parquet("data/onet_name_encoding.parquet")
char_to_token = encoding.set_index("BGI_ONET_NAME")["BGI_ONET_NAME_ENCODED"].to_dict()

EOT_TOKEN = 1016  # designate 1016 as the delimiting <END_OF_TEXT> token
char_to_token["<END_OF_TEXT>"] = EOT_TOKEN
token_to_char = {v: k for k, v in char_to_token.items()}

# # pre-tokenize all the splits one time up here
# test_tokens = [char_to_token[c] for c in open("data/test.txt", "r").read()]
# val_tokens = [char_to_token[c] for c in open("data/val.txt", "r").read()]
# train_tokens = [char_to_token[c] for c in open("data/train.txt", "r").read()]

with open("data/train_sequences.pkl", "rb") as f:
    train_tokens = pickle.load(f)
with open("data/val_sequences.pkl", "rb") as f:
    val_tokens = pickle.load(f)
with open("data/test_sequences.pkl", "rb") as f:
    test_tokens = pickle.load(f)

train_tokens = [item for sublist in train_tokens for item in sublist]
val_tokens = [item for sublist in val_tokens for item in sublist]
test_tokens = [item for sublist in test_tokens for item in sublist]

print("Reading data done")

# create the model
context_length = 3 # if 3 tokens predict the 4th, this is a 4-gram model
embedding_size = 1024
hidden_size = 4096
init_rng = RNG(1337)
# these two classes both produce the exact same results. One uses nn.Module the other doesn't.
# model = MLPRaw(vocab_size, context_length, embedding_size, hidden_size, init_rng)
model = MLP(vocab_size, context_length, embedding_size, hidden_size, init_rng)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Move model to device (for MLP which is nn.Module)
if isinstance(model, nn.Module):
    model = model.to(device)
# For MLPRaw, move all tensors to device
else:
    model.wte = model.wte.to(device)
    model.fc1_weights = model.fc1_weights.to(device)
    model.fc1_bias = model.fc1_bias.to(device)
    model.fc2_weights = model.fc2_weights.to(device)
    model.fc2_bias = model.fc2_bias.to(device)

# create the optimizer
learning_rate = 1e-3
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# training loop
timer = StepTimer()
batch_size = 2**16

# Calculate steps per epoch based on dataset size
tokens_per_epoch = len(train_tokens)
steps_per_epoch = tokens_per_epoch // batch_size
log_interval = max(1, int(steps_per_epoch * 0.1))  # Log 10 times per epoch
num_epochs = 10  # You can adjust this to control training duration
num_steps = steps_per_epoch * num_epochs

print(f'Dataset size: {tokens_per_epoch:,} tokens')
print(f'Steps per epoch: {steps_per_epoch:,} (with batch_size={batch_size})')
print(f'Logging interval: {log_interval} steps (0.1 epochs)')
print(f'Training for {num_epochs} epochs = {num_steps:,} steps')
print(f'num_steps {num_steps}, num_epochs {num_steps * batch_size / len(train_tokens):.2f}')
train_data_iter = dataloader(train_tokens, context_length, batch_size, device)

# Create checkpoint directory if it doesn't exist
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Initialize best validation loss tracking
best_val_loss = float('inf')
best_step = -1
best_model_path = None

for step in range(num_steps):
    # cosine learning rate schedule, from max lr to 0
    lr = learning_rate * 0.5 * (1 + math.cos(math.pi * step / num_steps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # every now and then evaluate the validation loss
    last_step = step == num_steps - 1
    if step % log_interval == 0 or last_step:
        train_loss = eval_split(model, train_tokens, max_batches=20, device=device)
        val_loss = eval_split(model, val_tokens, device=device)
        epoch_progress = (step % steps_per_epoch) / steps_per_epoch
        current_epoch = step // steps_per_epoch
        print(f'epoch {current_epoch:.1f} | step {step:6d} | train_loss {train_loss:.6f} | val_loss {val_loss:.6f} | lr {lr:e} | time/step {timer.get_dt()*1000:.4f}ms')
        
        # Save checkpoint if this is the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_step = step
            
            # Create checkpoint
            if isinstance(model, MLPRaw):
                model_type = "mlp_raw"
                # For MLPRaw, save the parameters
                save_dict = {
                    'wte': model.wte,
                    'fc1_weights': model.fc1_weights,
                    'fc1_bias': model.fc1_bias,
                    'fc2_weights': model.fc2_weights,
                    'fc2_bias': model.fc2_bias,
                    'params': {
                        'vocab_size': vocab_size,
                        'context_length': context_length,
                        'embedding_size': embedding_size,
                        'hidden_size': hidden_size
                    },
                    'step': step,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss
                }
            else:
                model_type = "mlp_module"
                # For nn.Module models, save the state_dict
                save_dict = {
                    'model_state_dict': model.state_dict(),
                    'params': {
                        'vocab_size': vocab_size,
                        'context_length': context_length,
                        'embedding_size': embedding_size,
                        'hidden_size': hidden_size
                    },
                    'step': step,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss
                }
            
            # Use a consistent filename for the best model, overwriting previous best
            best_model_filename = f"{checkpoint_dir}/best_model_{model_type}.pt"
            torch.save(save_dict, best_model_filename)
            best_model_path = best_model_filename
            # print(f"New best model saved to {best_model_filename} (val_loss: {val_loss:.6f}, step: {step})")
            
            # Log the checkpoint in the tracking file
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_file = "model_tracking.csv"
            log_exists = os.path.exists(log_file)
            
            with open(log_file, mode='a', newline='') as f:
                fieldnames = ['timestamp', 'model_type', 'context_length', 'embedding_size', 
                            'hidden_size', 'learning_rate', 'train_loss', 'val_loss', 
                            'step', 'is_best', 'filename']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                if not log_exists:
                    writer.writeheader()
                
                writer.writerow({
                    'timestamp': timestamp,
                    'model_type': model_type,
                    'context_length': context_length,
                    'embedding_size': embedding_size,
                    'hidden_size': hidden_size,
                    'learning_rate': learning_rate,
                    'train_loss': f"{train_loss:.6f}",
                    'val_loss': f"{val_loss:.6f}",
                    'step': step,
                    'is_best': True,
                    'filename': best_model_filename
                })
        
        # Also save regular checkpoints at the end of each epoch
        current_epoch = step // steps_per_epoch
        next_step = (step + 1) // steps_per_epoch
        if current_epoch != next_step or last_step:  # If we're changing epochs or at the last step
            checkpoint_filename = f"{checkpoint_dir}/checkpoint_{model_type}_epoch{current_epoch}.pt"
            if isinstance(model, MLPRaw):
                checkpoint_dict = {
                    'wte': model.wte,
                    'fc1_weights': model.fc1_weights,
                    'fc1_bias': model.fc1_bias,
                    'fc2_weights': model.fc2_weights,
                    'fc2_bias': model.fc2_bias,
                    'params': {
                        'vocab_size': vocab_size,
                        'context_length': context_length,
                        'embedding_size': embedding_size,
                        'hidden_size': hidden_size
                    },
                    'step': step,
                    'epoch': current_epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss
                }
            else:
                checkpoint_dict = {
                    'model_state_dict': model.state_dict(),
                    'params': {
                        'vocab_size': vocab_size,
                        'context_length': context_length,
                        'embedding_size': embedding_size,
                        'hidden_size': hidden_size
                    },
                    'step': step,
                    'epoch': current_epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss
                }
            torch.save(checkpoint_dict, checkpoint_filename)
            print(f"Epoch {current_epoch} completed - Checkpoint saved to {checkpoint_filename}")

    # training step
    with timer:
        # get the next batch of training data
        inputs, targets = next(train_data_iter)
        # forward pass (calculate the loss)
        logits, loss = model(inputs, targets)
        # backpropagate pass (calculate the gradients)
        loss.backward()
        # step the optimizer (update the parameters)
        optimizer.step()
        optimizer.zero_grad()

# model inference
# hardcode a prompt from which we'll continue the text
# sample_rng = RNG(42)
# prompt = "\nrichard"
# context = [char_to_token[c] for c in prompt]
# assert len(context) >= context_length
# context = context[-context_length:] # crop to context_length
# print(prompt, end='', flush=True)
# # now let's sample 200 more tokens that follow
# with torch.inference_mode():
#     for _ in range(200):
#         # take the last context_length tokens and predict the next one
#         context_tensor = torch.tensor(context).unsqueeze(0) # (1, T)
#         logits, _ = model(context_tensor) # (1, V)
#         probs = softmax(logits[0]) # (V, )
#         coinf = sample_rng.random() # "coin flip", float32 in range [0, 1)
#         next_token = sample_discrete(probs, coinf)
#         context = context[1:] + [next_token] # update the token tape
#         print(token_to_char[next_token], end='', flush=True)
# print() # newline

# and finally report the test loss
test_loss = eval_split(model, test_tokens, device=device)
print(f'test_loss {test_loss}')

# Save the trained model
model_params = f"context{context_length}_emb{embedding_size}_hidden{hidden_size}"
if isinstance(model, MLPRaw):
    model_type = "mlp_raw"
    # For MLPRaw, save the parameters
    save_dict = {
        'wte': model.wte,
        'fc1_weights': model.fc1_weights,
        'fc1_bias': model.fc1_bias,
        'fc2_weights': model.fc2_weights,
        'fc2_bias': model.fc2_bias,
        'params': {
            'vocab_size': vocab_size,
            'context_length': context_length,
            'embedding_size': embedding_size,
            'hidden_size': hidden_size
        }
    }
else:
    model_type = "mlp_module"
    # For nn.Module models, save the state_dict
    save_dict = {
        'model_state_dict': model.state_dict(),
        'params': {
            'vocab_size': vocab_size,
            'context_length': context_length,
            'embedding_size': embedding_size,
            'hidden_size': hidden_size
        }
    }

model_filename = f"model_{model_type}_{model_params}_loss{test_loss:.4f}.pt"
torch.save(save_dict, model_filename)
print(f"Model saved to {model_filename}")

# Log parameters and results to a tracking file
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log_file = "model_tracking.csv"
log_exists = os.path.exists(log_file)

with open(log_file, mode='a', newline='') as f:
    fieldnames = ['timestamp', 'model_type', 'context_length', 'embedding_size', 
                  'hidden_size', 'learning_rate', 'train_loss', 'val_loss', 
                  'test_loss', 'model_filename', 'num_steps', 'best_val_loss', 'best_step']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    
    if not log_exists:
        writer.writeheader()
    
    # Get final train and val loss
    final_train_loss = eval_split(model, train_tokens, max_batches=20, device=device)
    final_val_loss = eval_split(model, val_tokens, device=device)
    
    writer.writerow({
        'timestamp': timestamp,
        'model_type': model_type,
        'context_length': context_length,
        'embedding_size': embedding_size,
        'hidden_size': hidden_size,
        'learning_rate': learning_rate,
        'train_loss': f"{final_train_loss:.6f}",
        'val_loss': f"{final_val_loss:.6f}",
        'test_loss': f"{test_loss:.6f}",
        'model_filename': model_filename,
        'num_steps': num_steps,
        'best_val_loss': f"{best_val_loss:.6f}",
        'best_step': best_step
    })

print(f"Training info logged to {log_file}")
print(f"Best validation loss: {best_val_loss:.6f} at step {best_step}")
