import torch
from tqdm import tqdm


class Trainer:
    """Utility class for training feed-forward neural networks."""

    def __init__(
        self,
        model,
        optimizer_type: str = "adam",
        learning_rate: float = 0.001,
        epoch: int = 100,
        batch_size: int = 64,
    ):
        self.model = model
        if optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(
                model.parameters(), lr=learning_rate, momentum=0.9
            )
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.epoch = epoch
        self.batch_size = batch_size
        self.loss = torch.nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, train_loader, val_loader=None, silent: bool = False):
        """Train the network for a fixed number of epochs."""
        self.model.to(self.device)
        history = []
        for _ in tqdm(range(self.epoch), disable=silent):
            self.model.train()
            epoch_loss = 0.0
            for feats, labels in train_loader:
                feats, labels = feats.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(feats)
                loss = self.loss(out, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            history.append(epoch_loss)
        return history

    def evaluate(self, data_loader):
        """Return the average loss on ``data_loader``."""
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for feats, labels in data_loader:
                feats, labels = feats.to(self.device), labels.to(self.device)
                pred = self.model(feats)
                total += self.loss(pred, labels).item()
        return total
