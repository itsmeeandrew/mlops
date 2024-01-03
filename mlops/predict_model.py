import torch
import click
from mlops.models.model import NeuralNet
import torch.utils.data as data_utils

@click.command()
@click.argument("model_checkpoint")
@click.argument("data_path")
def predict(model_checkpoint, data_path):
    model = NeuralNet()
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()
    
    images = torch.load(data_path)
    dataset = data_utils.TensorDataset(images)
    dataloader = data_utils.DataLoader(dataset, batch_size=64, shuffle=False)

    with torch.no_grad():
        prediction = torch.cat([model(images) for images, in dataloader])

    torch.save(prediction, f"mlops/predictions/predictions.pt")
    print("Successfully saved predictions.")

    return prediction

if __name__ == '__main__':
    try:
        predict()
    except Exception as e:
        print("Error while predicting.")
        print(e)