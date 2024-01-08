from mlops.train_model import load_dataset
import pytest
import os.path
from tests import _PATH_DATA


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_length():
    trainloader, testloader = load_dataset(1)

    assert len(trainloader) == 50000, "Length of trainloader is not as expected"
    assert len(testloader) == 5000, "Length of testloader is not as expected"

@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_shape():
    trainloader, testloader = load_dataset(1)

    for images, _ in trainloader:
        assert images.shape == (1, 28, 28), "Shape of images is not as expected in trainloader"

    for images, _ in testloader:
        assert images.shape == (1, 28, 28), "Shape of images is not as expected in testloader"