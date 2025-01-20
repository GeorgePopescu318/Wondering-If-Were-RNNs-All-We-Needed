import torch
import pickle

# Define the LSTMbyHand class to match the saved model structure
class LSTMbyHand(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size=128, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)

        # Parameters for each LSTM layer
        self.wlr = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(hidden_size, hidden_size)) for _ in range(num_layers)]
        )
        self.wpr = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(hidden_size, hidden_size)) for _ in range(num_layers)]
        )
        self.wcs = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(hidden_size, hidden_size)) for _ in range(num_layers)]
        )

        self.blr = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros(hidden_size)) for _ in range(num_layers)]
        )
        self.bpr = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros(hidden_size)) for _ in range(num_layers)]
        )
        self.bcs = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros(hidden_size)) for _ in range(num_layers)]
        )

        self.fc_out = torch.nn.Linear(hidden_size, vocab_size)

    def lstm_unit(self, input_value, long_memory, layer_idx):
        """LSTM calculations for a single layer."""
        long_remember_percent = torch.sigmoid(
            torch.matmul(input_value, self.wlr[layer_idx]) + self.blr[layer_idx]
        )
        potential_remember_percent = torch.sigmoid(
            torch.matmul(input_value, self.wpr[layer_idx]) + self.bpr[layer_idx]
        )
        candid_hidden_state = torch.tanh(
            torch.matmul(input_value, self.wcs[layer_idx]) + self.bcs[layer_idx]
        )

        updated_long_memory = (long_remember_percent * long_memory) + (
            potential_remember_percent * candid_hidden_state
        )
        return updated_long_memory

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.size()
        embedded = self.embedding(input_seq)  # [batch_size, seq_len, hidden_size]

        # Initialize hidden and cell states for all layers
        long_memory = [
            torch.zeros(batch_size, self.hidden_size, device=input_seq.device) for _ in range(self.num_layers)
        ]

        outputs = []
        for t in range(seq_len):
            x = embedded[:, t]
            for layer_idx in range(self.num_layers):
                x = self.lstm_unit(x, long_memory[layer_idx], layer_idx)
                long_memory[layer_idx] = x
            outputs.append(self.fc_out(x))

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
    filename = "lstm_model.pkl"  # Adjust this if your filename differs
    model, char_to_idx, idx_to_char = load_model(filename)

    # Generate text starting with a given prompt
    start_text = "SEBASTIAN :\nWhy\nDoth it not then our eyelids sink? I find not\nMyself disposed to sleep.\n"  # Modify the starting text as needed
    generated_text = generate_text(model, start_text, char_to_idx, idx_to_char, length=50)

    # Print the generated text
    print("Generated Text:")
    print(generated_text)
