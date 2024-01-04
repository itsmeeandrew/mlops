import click
import torch
import torch.utils.data as data_utils

from mlops.models.model import NeuralNet


@click.command()
@click.argument("model_checkpoint")
@click.argument("data_path")
def predict(model_checkpoint, data_path):
    """
    Predicts on the provided data and saves the predictions to mlops/predictions/predictions.pt

    Parameters:
        model_checkpoint: Path to model checkpoint
        data_path: Path pre-loaded data (B x H x W)

    Returns:
        predictions: Predictions from the model (B x 10)
    """
    model = NeuralNet()
    model.load_state_dict(torch.load(model_checkpoint, map_location=torch.device("cpu")))
    model.eval()

    images = torch.load(data_path)
    dataset = data_utils.TensorDataset(images)
    dataloader = data_utils.DataLoader(dataset, batch_size=64, shuffle=False)

    with torch.no_grad():
        prediction = torch.cat([model(images) for (images,) in dataloader])

    torch.save(prediction, "mlops/predictions/predictions.pt")
    print("Successfully saved predictions.")

    return prediction


if __name__ == "__main__":
    try:
        predict()
    except Exception as e:
        print("Error while predicting.")
        print(e)
