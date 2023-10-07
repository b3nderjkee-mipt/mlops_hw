import os
import time

import torch
from data.config import CNN_CONFIG, CNN_OPTIMIZER_CONFIG, CNN_TRAIN_CONFIG
from data.dataset import mnist_dataload
from data.train_evaluate import train
from model import ModelCNN


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size_cnn = CNN_TRAIN_CONFIG["batch_size"]
    n_epochs_cnn = CNN_TRAIN_CONFIG["n_epochs"]

    train_loader_cnn = mnist_dataload(batch_size=batch_size_cnn, train=True, shuffle=True)

    cnn = ModelCNN(**CNN_CONFIG).to(device)

    optimizer_cnn = torch.optim.Adam(cnn.parameters(), **CNN_OPTIMIZER_CONFIG)

    start_time = time.time()
    print("CNN training...")
    train(cnn, optimizer_cnn, train_loader_cnn, device)

    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")

    torch.save(cnn.state_dict(), "saved_models/cnn.pt")
    print("Model successfully saved.")
