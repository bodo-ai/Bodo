def torch_import_guard():
    try:
        import torch  # noqa: F401
    except ImportError:
        raise ImportError(
            "PyTorch is not installed. Please install it to use TorchTrainer."
        )


class TorchTrainer:
    def __init__(self, model, optimizer, loss_fn, device="cpu"):
        torch_import_guard()

        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train_step(self, x_batch, y_batch):
        torch_import_guard()
        import torch

        self.model.train()
        x_batch = torch.tensor(x_batch, dtype=torch.float32).to(self.device)
        y_batch = torch.tensor(y_batch, dtype=torch.float32).to(self.device)

        self.optimizer.zero_grad()
        y_pred = self.model(x_batch)
        loss = self.loss_fn(y_pred, y_batch)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, x_val, y_val):
        torch_import_guard()
        import torch

        self.model.eval()
        x_val = torch.tensor(x_val, dtype=torch.float32).to(self.device)
        y_val = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            y_pred = self.model(x_val)
            loss = self.loss_fn(y_pred, y_val)

        return loss.item()
