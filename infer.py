import os

import hydra
import torch
from mlops_hw.config import Params
from mlops_hw.dataset import mnist_dataload
from mlops_hw.train_evaluate import evaluate_model
from model import ModelCNN


@hydra.main(config_path="configs", config_name="model_config", version_base="1.3")
def infer(cfg: Params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = cfg.training.batch_size
    eval_loader = mnist_dataload(batch_size=batch_size, train=False, shuffle=True)
    cnn = ModelCNN(**cfg.model).to(device)
    cnn.load_state_dict(torch.load("saved_models/cnn.pt"))

    cnn_loss, cnn_accuracy = evaluate_model(cnn, eval_loader, device)
    print(f"CNN eval accuracy = {cnn_accuracy:.4f}")

    if not os.path.exists("reports"):
        os.makedirs("reports")

    with open("reports/inf_report.txt", "w") as report:
        report.write(f"CNN eval loss = {cnn_loss}")
        report.write("\n")
        report.write(f"CNN eval accuracy = {cnn_accuracy}")
    print("Report successfully saved.")


if __name__ == "__main__":
    infer()
