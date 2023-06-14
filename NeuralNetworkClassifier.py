import numpy as np

class NeuralNetworkClassifier:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weights1 = np.random.randn(hidden_size, input_size)
        self.weights2 = np.random.randn(output_size, hidden_size)

    def forward(self, inputs):
        hidden_layer = np.dot(self.weights1, inputs)
        hidden_layer_activation = self.sigmoid(hidden_layer)

        output_layer = np.dot(self.weights2, hidden_layer_activation)
        output_layer_activation = self.sigmoid(output_layer)

        return output_layer_activation

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

def main():
    # Initialize the neural network classifier
    input_size = 4
    hidden_size = 5
    output_size = 3
    classifier = NeuralNetworkClassifier(input_size, hidden_size, output_size)

    # Example input
    inputs = np.array([0.5, 0.2, 0.1, 0.8])

    # Perform forward propagation
    predicted_probs = classifier.forward(inputs)

    # Print the predicted probabilities
    print("Predicted Probabilities:")
    print(predicted_probs)

if __name__ == "__main__":
    main()
