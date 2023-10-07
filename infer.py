import os
import time

import torch
from data.config import CNN_CONFIG, CNN_TRAIN_CONFIG
from data.dataset import mnist_dataload
from data.train_evaluate import evaluate
from model import ModelCNN


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size_cnn = CNN_TRAIN_CONFIG["batch_size"]
    batch_size = batch_size_cnn // 2

    eval_loader = mnist_dataload(batch_size=batch_size, train=False, shuffle=True)

    cnn = ModelCNN(**CNN_CONFIG).to(device)

    cnn.load_state_dict(torch.load("saved_models/cnn.pt"))

    start_time = time.time()
    cnn_loss, cnn_accuracy = evaluate(cnn, eval_loader, device)
    print(f"CNN eval accuracy = {cnn_accuracy:.4f}")

    if not os.path.exists("reports"):
        os.makedirs("reports")

    with open("reports/inf_report.txt", "w") as report:
        report.write(f"CNN eval loss = {cnn_loss}")
        report.write("\n")
        report.write(f"CNN eval accuracy = {cnn_accuracy}")
    print("Report successfully saved.")
