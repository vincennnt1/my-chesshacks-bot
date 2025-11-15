import numpy as np

class TinyChessModelNumpy:
    def __init__(self, weights_path):
        data = np.load(weights_path)

        # Load layers
        self.w1 = data["net.0.weight"]
        self.b1 = data["net.0.bias"]

        self.w2 = data["net.2.weight"]
        self.b2 = data["net.2.bias"]

        self.w3 = data["net.4.weight"]
        self.b3 = data["net.4.bias"]

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        # x shape: (832,)
        x = x.astype(np.float32)

        # Layer 1
        x = self.relu(self.w1 @ x + self.b1)

        # Layer 2
        x = self.relu(self.w2 @ x + self.b2)

        # Layer 3
        x = self.w3 @ x + self.b3

        return float(x[0])
