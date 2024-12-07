# Simple Neural Network 
## XOR Neural Network from Scratch

This repository contains a simple implementation of a feed-forward neural network built from scratch in Python. The network is designed to solve the XOR problem, a classic challenge in machine learning.

## Features
- Implements a fully connected neural network with:
  - 2 input neurons
  - 4 hidden neurons
  - 1 output neuron
- Uses sigmoid activation functions.
- Trains the network using backpropagation and gradient descent.
- Demonstrates forward pass, error calculation, and weight updates.

## Getting Started

### Prerequisites
- Python 3.7+
- NumPy

Install the required library using:
```bash
pip install numpy 
```

## Running the Code
### Clone the repository:
``` bash
git clone https://github.com/yourusername/xor-neural-network.git
cd xor-neural-network
```

### Run the script:
```
python xor_neural_network.py
```

### Output
After training, the network will output the predicted results for the XOR problem, similar to this:
```
Input: [0 0], Output predicted: 0.0123, Expected: 0
Input: [0 1], Output predicted: 0.9867, Expected: 1
Input: [1 0], Output predicted: 0.9865, Expected: 1
Input: [1 1], Output predicted: 0.0139, Expected: 0
```

## How It Works

### Forward Pass: 

Calculates the outputs of each layer using the sigmoid activation function.

### Backpropagation: 

Updates weights and biases by propagating the error backward through the network.

### Training: 

Iteratively adjusts the weights and biases over multiple epochs to minimize the error.


## License

This project is licensed under the MIT License.
