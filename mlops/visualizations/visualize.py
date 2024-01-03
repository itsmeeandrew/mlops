import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

from mlops.models.model import NeuralNet


def intermediate():
    params = torch.load("mlops\checkpoints\checkpoint_70.pth")
    model = NeuralNet()
    model.load_state_dict(params)
    model.train()

    embedded = TSNE(n_components=2, perplexity=5).fit_transform(model.fc4.weight)

    plt.scatter(embedded[:, 0], embedded[:, 1])
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("2D representations of weights in fc4 layer")
    plt.savefig("reports/figures/intermediate.png")

    print("Intermediate representations saved.")


if __name__ == "__main__":
    try:
        intermediate()
    except Exception as e:
        print("Error while creating intermediate representations.")
        print(e)
