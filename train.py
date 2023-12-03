import os

import hydra
import mlflow
import torch
from mlops_hw.config import Params
from mlops_hw.dataset import mnist_dataload
from mlops_hw.train_evaluate import train_model
from model import ModelCNN


def convert_onnx(model, save_path, input_size):
    model.eval()
    dummy_input = torch.randn((1,) + input_size, requires_grad=True)
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
    )
    print(f"Model has been converted to ONNX and saved in {save_path}")


@hydra.main(config_path="configs", config_name="model_config", version_base="1.3")
def train(cfg: Params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = cfg.training.batch_size
    n_epochs = cfg.training.n_epochs
    train_loader = mnist_dataload(batch_size=batch_size, train=True, shuffle=True)
    cnn = ModelCNN(**cfg.model).to(device)
    optimizer_cnn = torch.optim.Adam(cnn.parameters(), lr=cfg.training.learning_rate)

    remote_server_uri = "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.pytorch.autolog()
    # start_time = time.time()
    print("CNN training...")
    with mlflow.start_run() as _:
        loss, acc = train_model(cnn, n_epochs, optimizer_cnn, train_loader, device)
        mlflow.log_params(cfg.model)
        mlflow.log_params(cfg.training)
        mlflow.log_metric("train_loss", loss)
        mlflow.log_metric("train_acc", acc)

    input_size = next(iter(train_loader))[0].shape[1:]

    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")

    torch.save(cnn.state_dict(), "saved_models/cnn.pt")
    print("Model successfully saved.")

    convert_onnx(cnn.eval().cpu(), "saved_models/cnn.onnx", input_size)


if __name__ == "__main__":
    train()
