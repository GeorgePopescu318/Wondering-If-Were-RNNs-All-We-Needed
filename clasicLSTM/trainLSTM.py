import lightning as L
from lightning.pytorch.loggers import CSVLogger  # Import CSVLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch.accelerators import find_usable_cuda_devices
import pickle


# Save the model to a pickle file
def save_model(model, dataset, filename="lstm_model.pkl"):
    model_data = {
        "model_state_dict": model.state_dict(),
        "vocab_size": dataset.vocab_size,
        "char_to_idx": dataset.char_to_idx,
        "idx_to_char": dataset.idx_to_char,
    }
    with open(filename, "wb") as f:
        pickle.dump(model_data, f)
    print(f"Model saved to {filename}")


# Load the model from a pickle file
def load_model(filename="lstm_model.pkl"):
    with open(filename, "rb") as f:
        model_data = pickle.load(f)
    
    vocab_size = model_data["vocab_size"]
    model = LSTMbyHand(vocab_size)
    model.load_state_dict(model_data["model_state_dict"])
    model.eval()

    # Restore the character mappings
    char_to_idx = model_data["char_to_idx"]
    idx_to_char = model_data["idx_to_char"]

    print(f"Model loaded from {filename}")
    return model, char_to_idx, idx_to_char


# Create a character-level dataset
class CharDataset(Dataset):
    def __init__(self, text, seq_length):
        self.chars = sorted(list(set(text)))  # Unique characters
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

        self.data = [self.char_to_idx[char] for char in text]  # Convert text to indices
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        input_seq = self.data[idx : idx + self.seq_length]
        target_seq = self.data[idx + 1 : idx + self.seq_length + 1]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)


# Custom multilayer LSTM by hand
class LSTMbyHand(L.LightningModule):
    def __init__(self, vocab_size, hidden_size=128, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # Parameters for each gate and layer
        self.W_f = nn.ParameterList([nn.Parameter(torch.randn(hidden_size, hidden_size)) for _ in range(num_layers)])
        self.U_f = nn.ParameterList([nn.Parameter(torch.randn(hidden_size, hidden_size)) for _ in range(num_layers)])
        self.b_f = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size)) for _ in range(num_layers)])

        self.W_i = nn.ParameterList([nn.Parameter(torch.randn(hidden_size, hidden_size)) for _ in range(num_layers)])
        self.U_i = nn.ParameterList([nn.Parameter(torch.randn(hidden_size, hidden_size)) for _ in range(num_layers)])
        self.b_i = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size)) for _ in range(num_layers)])

        self.W_c = nn.ParameterList([nn.Parameter(torch.randn(hidden_size, hidden_size)) for _ in range(num_layers)])
        self.U_c = nn.ParameterList([nn.Parameter(torch.randn(hidden_size, hidden_size)) for _ in range(num_layers)])
        self.b_c = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size)) for _ in range(num_layers)])

        self.W_o = nn.ParameterList([nn.Parameter(torch.randn(hidden_size, hidden_size)) for _ in range(num_layers)])
        self.U_o = nn.ParameterList([nn.Parameter(torch.randn(hidden_size, hidden_size)) for _ in range(num_layers)])
        self.b_o = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size)) for _ in range(num_layers)])

        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.size()
        embedded = self.embedding(input_seq)  # [batch_size, seq_len, hidden_size]

        # Initialize hidden and cell states for all layers
        h = [
            torch.zeros(batch_size, self.hidden_size, device=self.device) for _ in range(self.num_layers)
        ]
        c = [
            torch.zeros(batch_size, self.hidden_size, device=self.device) for _ in range(self.num_layers)
        ]

        outputs = []
        for t in range(seq_len):
            x = embedded[:, t]
            for layer_idx in range(self.num_layers):
                # Classic LSTM calculations for each gate
                forget_gate = torch.sigmoid(torch.matmul(x, self.W_f[layer_idx]) + 
                                            torch.matmul(h[layer_idx], self.U_f[layer_idx]) + 
                                            self.b_f[layer_idx])
                input_gate = torch.sigmoid(torch.matmul(x, self.W_i[layer_idx]) + 
                                           torch.matmul(h[layer_idx], self.U_i[layer_idx]) + 
                                           self.b_i[layer_idx])
                candidate_cell = torch.tanh(torch.matmul(x, self.W_c[layer_idx]) + 
                                            torch.matmul(h[layer_idx], self.U_c[layer_idx]) + 
                                            self.b_c[layer_idx])
                output_gate = torch.sigmoid(torch.matmul(x, self.W_o[layer_idx]) + 
                                            torch.matmul(h[layer_idx], self.U_o[layer_idx]) + 
                                            self.b_o[layer_idx])

                # Update cell state and hidden state
                c[layer_idx] = forget_gate * c[layer_idx] + input_gate * candidate_cell
                h[layer_idx] = output_gate * torch.tanh(c[layer_idx])

                # Pass hidden state to the next layer
                x = h[layer_idx]

            # Append output from the top layer
            outputs.append(self.fc_out(h[-1]))

        return torch.stack(outputs, dim=1)  # [batch_size, seq_len, vocab_size]

    def training_step(self, batch, batch_idx):
        input_seq, target_seq = batch
        predictions = self(input_seq)  # [batch_size, seq_len, vocab_size]
        loss = F.cross_entropy(
            predictions.view(-1, self.vocab_size), target_seq.view(-1)
        )
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)

# Preparing Shakespeare data
def prepare_data(text, seq_length=30):
    dataset = CharDataset(text, seq_length)
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=True)
    return dataset, dataloader


# Example usage
with open("input.txt", "r") as f:
    text = f.read()

seq_length = 100
dataset, dataloader = prepare_data(text, seq_length)

# Initialize model
vocab_size = dataset.vocab_size
model = LSTMbyHand(vocab_size, hidden_size=128, num_layers=2)

# Create a CSV logger
csv_logger = CSVLogger(save_dir="logs", name="lstm_training")

# Train the model
trainer = L.Trainer(max_epochs=1, accelerator="cuda", devices=find_usable_cuda_devices(1), logger=csv_logger)
trainer.fit(model, train_dataloaders=dataloader)

# Save the trained model
save_model(model, dataset, filename="lstm_model.pkl")