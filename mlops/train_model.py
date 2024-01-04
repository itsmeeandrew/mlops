import matplotlib.pyplot as plt
import torch
import torch.utils.data as data_utils

from mlops.models.small_model import NeuralNet


def load_dataset():
    train_images = torch.load("data/processed/train_images.pt")
    train_targets = torch.load("data/processed/train_targets.pt")
    train = data_utils.TensorDataset(train_images, train_targets)
    trainloader = data_utils.DataLoader(train, batch_size=64, shuffle=True)

    test_images = torch.load("data/processed/test_images.pt")
    test_targets = torch.load("data/processed/test_targets.pt")
    test = data_utils.TensorDataset(test_images, test_targets)
    testloader = data_utils.DataLoader(test, batch_size=64, shuffle=False)

    return trainloader, testloader


def save_loss_plot(train_losses, test_losses):
    plt.plot(train_losses, label="Training loss")
    plt.plot(test_losses, label="Test loss")
    plt.legend()
    plt.title("Losses")
    plt.savefig("reports/figures/loss.png")
    print("Loss plot saved.")


def save_model(model, epoch):
    torch.save(model.state_dict(), f"mlops/checkpoints/checkpoint_{epoch}.pth")
    print("Model saved.")


def train():
    """
    Trains the model and saves the trained model to mlops/checkpoints/checkpoint_[epoch].pth and saves the loss plot to reports/figures/loss.png
    """

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", DEVICE)

    model = NeuralNet()
    model.to(DEVICE)

    trainloader, testloader = load_dataset()

    criterion = torch.nn.NLLLoss(reduction="sum").to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    epochs = 1
    train_losses = []
    test_losses = []

    for e in range(epochs):
        print(f"{e+1}/{epochs}")
        model.train()
        running_loss = 0
        for images, targets in trainloader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            print(f"Training loss: {running_loss/len(trainloader):.3f}")
            train_losses.append(running_loss / len(trainloader))

        running_loss = 0
        for images, targets in testloader:
            model.eval()

            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            with torch.no_grad():
                output = model(images)
                loss = criterion(output, targets)
                running_loss += loss.item()
        else:
            print(f"Test loss: {running_loss/len(testloader):.3f}")
            test_losses.append(running_loss / len(testloader))

    save_model(model, e)
    save_loss_plot(train_losses, test_losses)


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print("Error during training.")
        print(e)
