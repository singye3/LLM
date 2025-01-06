import torch.nn as nn
class PositionWiseFeedForward(nn.Module):
    """
    Implements the Position-wise Feed-Forward Neural Network used in Transformer models.

    Args:
        d_model (int): The dimension of the input and output features.
        d_ff (int): The dimension of the hidden layer.

    Attributes:
        fc1 (nn.Linear): The first fully connected layer.
        fc2 (nn.Linear): The second fully connected layer.
        relu (nn.ReLU): The ReLU activation function.

    Methods:
        forward(x):
            Passes the input tensor through the feed-forward network.
            Args:
                x (torch.Tensor): The input tensor of shape (batch_size, seq_length, d_model).
            Returns:
                torch.Tensor: The output tensor of shape (batch_size, seq_length, d_model).
    """
    def __init__(self, d_model: int, d_ff: int):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))