from mlops.models.model import NeuralNet
import torch
import pytest

def test_forward():
    net = NeuralNet()
    assert net(torch.zeros(2, 28, 28)).shape == (2, 10), "Shape is not as expected"

def test_error_on_wrong_shape():
    net = NeuralNet()
    with pytest.raises(ValueError, match='Batch size must be at least 2, because of batch normalization'):
        net(torch.zeros(1, 28, 28))

if __name__ == "__main__":
    test_forward()