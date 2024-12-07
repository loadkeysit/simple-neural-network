import numpy as np

# Configurazione della rete neurale
input_neurons = 2
hidden_neurons = 4
output_neurons = 1

# Inizializza pesi e bias
np.random.seed(42)
weights_input_hidden = np.random.randn(input_neurons, hidden_neurons)  # (2, 4)
weights_hidden_output = np.random.randn(hidden_neurons, output_neurons)  # (4, 1)
bias_hidden = np.random.randn(hidden_neurons)  # (4,)
bias_output = np.random.randn(output_neurons)  # (1,)

# Dati di input e output attesi
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_outputs = np.array([[0], [1], [1], [0]])

# Funzioni di attivazione
def sigmoid(x):
    """Funzione sigmoid."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivata della funzione sigmoid."""
    return x * (1 - x)

# Forward pass
def forward_pass(input_data):
    """Calcola il forward pass della rete."""
    hidden_layer_input = np.dot(input_data, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    output_layer_output = sigmoid(output_layer_input)
    return hidden_layer_output, output_layer_output

# Backpropagation
def backpropagation(input_data, hidden_output, output_output, target_output):
    """Esegue il backpropagation per aggiornare i pesi e i bias."""
    global weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

    # Calcola l'errore dell'output
    output_error = target_output - output_output
    output_delta = output_error * sigmoid_derivative(output_output)

    # Propaga l'errore allo strato nascosto
    hidden_error = np.dot(output_delta, weights_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_output)

    # Aggiorna i pesi e i bias
    weights_hidden_output += np.dot(hidden_output.T, output_delta) * learning_rate
    bias_output += np.sum(output_delta, axis=0) * learning_rate
    weights_input_hidden += np.dot(input_data.T, hidden_delta) * learning_rate
    bias_hidden += np.sum(hidden_delta, axis=0) * learning_rate

# Parametri di training
learning_rate = 0.1
epochs = 10000

# Training della rete neurale
for epoch in range(epochs):
    for i in range(len(inputs)):
        # Forward pass
        input_layer = inputs[i].reshape(1, -1)
        hidden_layer_output, output_layer_output = forward_pass(input_layer)

        # Backpropagation
        backpropagation(input_layer, hidden_layer_output, output_layer_output, expected_outputs[i].reshape(1, -1))

# Test della rete neurale
print("Risultati finali:")
for i in range(len(inputs)):
    input_layer = inputs[i].reshape(1, -1)
    _, output_layer_output = forward_pass(input_layer)
    print(f"Input: {inputs[i]}, Output predetto: {output_layer_output.flatten()[0]:.4f}, Output atteso: {expected_outputs[i][0]}")
