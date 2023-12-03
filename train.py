import os

import hydra
import torch
from mlops_hw.config import Params
from mlops_hw.dataset import mnist_dataload
from mlops_hw.train_evaluate import train_model
from model import ModelCNN


# import time


@hydra.main(config_path="configs", config_name="model_config", version_base="1.3")
def train(cfg: Params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size_cnn = cfg.training.batch_size
    # n_epochs_cnn = cfg.training.n_epochs
    train_loader_cnn = mnist_dataload(batch_size=batch_size_cnn, train=True, shuffle=True)
    cnn = ModelCNN(**cfg.model).to(device)
    optimizer_cnn = torch.optim.Adam(cnn.parameters(), lr=cfg.training.learning_rate)

    # start_time = time.time()
    print("CNN training...")
    train_model(cnn, optimizer_cnn, train_loader_cnn, device)

    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")

    torch.save(cnn.state_dict(), "saved_models/cnn.pt")
    print("Model successfully saved.")


if __name__ == "__main__":
    train()
