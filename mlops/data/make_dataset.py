from glob import glob

import torch


def normalize_images(x):
    means = x.mean(dim=(1, 2))
    stds = x.std(dim=(1, 2))

    return (x - means[:, None, None]) / stds[:, None, None]


def load_and_save(prefix="train"):
    image_files = glob(f"data/raw/{prefix}_images*.pt")
    target_files = glob(f"data/raw/{prefix}_target*.pt")

    images = torch.cat([torch.load(f) for f in image_files])
    targets = torch.cat([torch.load(f) for f in target_files])

    images = normalize_images(images)

    torch.save(images, f"data/processed/{prefix}_images.pt")
    torch.save(targets, f"data/processed/{prefix}_targets.pt")


if __name__ == "__main__":
    try:
        load_and_save("train")
        load_and_save("test")
        print("Successfully created processed data.")
    except Exception as e:
        print("Error while creating processed data.")
        print(e)
