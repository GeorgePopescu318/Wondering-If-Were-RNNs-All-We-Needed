import torch
import pickle
import torch.nn as nn

# Define the LSTMbyHand class to match the saved model structure
class LSTMbyHand(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size=128, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)

        # Parameters for each LSTM layer
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

    def lstm_unit(self, input_value, h, c, layer_idx):
        """LSTM calculations for a single layer."""
        forget_gate = torch.sigmoid(
            torch.matmul(input_value, self.W_f[layer_idx]) +
            torch.matmul(h, self.U_f[layer_idx]) +
            self.b_f[layer_idx]
        )
        input_gate = torch.sigmoid(
            torch.matmul(input_value, self.W_i[layer_idx]) +
            torch.matmul(h, self.U_i[layer_idx]) +
            self.b_i[layer_idx]
        )
        candidate_cell = torch.tanh(
            torch.matmul(input_value, self.W_c[layer_idx]) +
            torch.matmul(h, self.U_c[layer_idx]) +
            self.b_c[layer_idx]
        )
        output_gate = torch.sigmoid(
            torch.matmul(input_value, self.W_o[layer_idx]) +
            torch.matmul(h, self.U_o[layer_idx]) +
            self.b_o[layer_idx]
        )

        # Update cell state and hidden state
        c = forget_gate * c + input_gate * candidate_cell
        h = output_gate * torch.tanh(c)

        return h, c

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.size()
        embedded = self.embedding(input_seq)  # [batch_size, seq_len, hidden_size]

        # Initialize hidden and cell states for all layers
        h = [torch.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]

        outputs = []
        for t in range(seq_len):
            x = embedded[:, t]
            for layer_idx in range(self.num_layers):
                h[layer_idx], c[layer_idx] = self.lstm_unit(x, h[layer_idx], c[layer_idx], layer_idx)
                x = h[layer_idx]  # Pass hidden state to the next layer

            # Append output from the top layer
            outputs.append(self.fc_out(h[-1]))

        return torch.stack(outputs, dim=1)  # [batch_size, seq_len, vocab_size]

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

# Generate text using the loaded model
def generate_text(model, start_text, char_to_idx, idx_to_char, length=100):
    model.eval()
    generated_text = start_text
    input_seq = torch.tensor(
        [char_to_idx[char] for char in start_text], dtype=torch.long
    ).unsqueeze(0)

    with torch.no_grad():
        for _ in range(length):
            predictions = model(input_seq)  # [1, seq_length, vocab_size]
            last_prediction = predictions[0, -1]  # [vocab_size]
            next_char_idx = torch.argmax(last_prediction).item()
            next_char = idx_to_char[next_char_idx]

            generated_text += next_char
            input_seq = torch.cat(
                [input_seq[:, 1:], torch.tensor([[next_char_idx]])], dim=1
            )

    return generated_text

# Main script
if __name__ == "__main__":
    # Load the trained model
    filename = "lstm_model_50_epics_LSTM.pkl"  # Adjust this if your filename differs
    model, char_to_idx, idx_to_char = load_model(filename)

    # Generate text starting with a given prompt
    start_text = "SEBASTIAN: Please you, sir, Do not omit the heavy offer of it: It seldom visits sorrow; when it doth, It is a comforter. \nANTONIO:"  # Modify the starting text as needed
    generated_text = generate_text(model, start_text, char_to_idx, idx_to_char, length=100)

    # Print the generated text
    print("Generated Text:")
    print(generated_text)