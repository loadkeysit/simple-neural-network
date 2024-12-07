import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Configura la rete neurale
input_neurons = 2
hidden_neurons = 4
output_neurons = 1

# Posizioni dei neuroni
input_positions = [(0, i) for i in range(input_neurons)]
hidden_positions = [(1, i) for i in range(hidden_neurons)]
output_positions = [(2, i) for i in range(output_neurons)]

# Pesi e bias
np.random.seed(42)
weights_input_hidden = np.random.randn(input_neurons, hidden_neurons)
weights_hidden_output = np.random.randn(hidden_neurons, output_neurons)
bias_hidden = np.random.randn(hidden_neurons)
bias_output = np.random.randn(output_neurons)

# Dati di input e output attesi
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_outputs = np.array([[0], [1], [1], [0]])

# Funzioni di attivazione
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Forward pass
def forward_pass(input_data):
    hidden_input = np.dot(input_data, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)
    output_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    output_output = sigmoid(output_input)
    return hidden_output, output_output

# Configurazione Matplotlib
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-1, 3)
ax.set_ylim(-1, max(input_neurons, hidden_neurons, output_neurons) + 1)
ax.axis('off')

# Disegna i neuroni
def draw_neurons(ax, positions, color, values=None):
    for i, pos in enumerate(positions):
        circle = plt.Circle(pos, 0.1, color=color, ec='black', lw=1)
        ax.add_artist(circle)
        if values is not None:
            ax.text(pos[0], pos[1], f"{values[i]:.2f}", ha='center', va='center', fontsize=8, color='white')

# Disegna le connessioni
def draw_connections(ax, start_positions, end_positions, weights):
    for i, start in enumerate(start_positions):
        for j, end in enumerate(end_positions):
            weight = weights[i, j]
            color = 'blue' if weight > 0 else 'red'
            alpha = min(abs(weight) / 2, 1)
            ax.plot([start[0], end[0]], [start[1], end[1]], color=color, alpha=alpha)

# Animazione
def update(frame):
    ax.clear()
    ax.set_xlim(-1, 3)
    ax.set_ylim(-1, max(input_neurons, hidden_neurons, output_neurons) + 1)
    ax.axis('off')

    # Input, hidden, output per il frame corrente
    input_data = inputs[frame % len(inputs)]
    hidden_output, output_output = forward_pass(input_data)

    # Disegna input, hidden, output
    draw_neurons(ax, input_positions, 'blue', input_data)
    draw_neurons(ax, hidden_positions, 'green', hidden_output)
    draw_neurons(ax, output_positions, 'red', output_output)

    # Disegna connessioni
    draw_connections(ax, input_positions, hidden_positions, weights_input_hidden)
    draw_connections(ax, hidden_positions, output_positions, weights_hidden_output)

# Creazione dell'animazione
ani = FuncAnimation(fig, update, frames=8, interval=1000, repeat=True)

# Salva l'animazione come GIF
ani.save("neural_network_animation.gif", writer='imagemagick')

# Per salvare come video MP4
# ani.save("neural_network_animation.mp4", writer='ffmpeg')

print("Animazione salvata come GIF!")
