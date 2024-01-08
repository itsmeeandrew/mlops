import hydra
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data_utils
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from mlops.models.model import NeuralNet


def load_dataset(batch_size):
    train_images = torch.load("data/processed/train_images.pt")
    train_targets = torch.load("data/processed/train_targets.pt")
    train = data_utils.TensorDataset(train_images, train_targets)
    trainloader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)

    test_images = torch.load("data/processed/test_images.pt")
    test_targets = torch.load("data/processed/test_targets.pt")
    test = data_utils.TensorDataset(test_images, test_targets)
    testloader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=False)

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


def get_model(cfg):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_hparams = cfg.hyperparameters
    model = NeuralNet(
        c1out=model_hparams["c1out"],
        c2out=model_hparams["c2out"],
        c3out=model_hparams["c3out"],
        fc1out=model_hparams["fc1out"],
        fc2out=model_hparams["fc2out"],
        fc3out=model_hparams["fc3out"],
        p_drop=model_hparams["p_drop"],
    )
    model.to(DEVICE)

    return model


def train(train_cfg, model_cfg):
    """
    Trains the model and saves the trained model to mlops/checkpoints/checkpoint_[epoch].pth and saves the loss plot to reports/figures/loss.png
    """

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", DEVICE)

    train_hparams = train_cfg.hyperparameters

    torch.manual_seed(train_hparams["seed"])

    model = get_model(model_cfg)

    trainloader, testloader = load_dataset(train_hparams["batch_size"])

    criterion = torch.nn.NLLLoss(reduction="sum").to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_hparams["lr"])
    epochs = train_hparams["epochs"]
    train_losses = []
    test_losses = []

    preds = []
    target = []
    for e in range(epochs):
        print(f"{e+1}/{epochs}")
        model.train()
        running_loss = 0
        for images, targets in trainloader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            output = model(images)

            preds.append(output.argmax(dim=1).cpu())
            target.append(targets.detach().cpu())

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

    target = torch.cat(target, dim=0)
    preds = torch.cat(preds, dim=0)

    report = classification_report(target, preds)
    with open("reports/figures/classification_report.txt", "w") as outfile:
        outfile.write(report)
    confmat = confusion_matrix(target, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=confmat)
    disp.plot()
    plt.savefig("reports/figures/confusion_matrix.png")

    # Currently do not work
    # save_model(model, e)
    # save_loss_plot(train_losses, test_losses)


def main():
    hydra.initialize(config_path="conf")
    train_cfg = hydra.compose("train_conf.yaml")
    model_cfg = hydra.compose("model_conf.yaml")
    train(train_cfg, model_cfg)


if __name__ == "__main__":
    """ try:
        main()
    except Exception as e:
        print("Error during training.")
        print(e) """
    main()
