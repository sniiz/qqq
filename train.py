import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard
import pandas as pd
import numpy as np
import tqdm


def fen_to_tensor(fen):
    """
    Convert a FEN string into a 8x8x14 tensor.
    """
    piece_symbols = "prnbkqPRNBKQ"
    piece_to_int = {piece: i for i, piece in enumerate(piece_symbols)}
    tensor = torch.zeros(14, 8, 8)

    fen_board, fen_info = fen.split(" ")[0:2]
    for row, fen_row in enumerate(fen_board.split("/")):
        col = 0
        for char in fen_row:
            if char.isdigit():
                col += int(char)
            else:
                tensor[piece_to_int[char], row, col] = 1
                col += 1

    if fen_info[0] == "w":
        tensor[12, :, :] = 1
    else:
        tensor[13, :, :] = 1

    if "K" in fen_info:
        tensor[12, 0, 4] = 1
    if "Q" in fen_info:
        tensor[12, 0, 0] = 1
    if "k" in fen_info:
        tensor[13, 7, 4] = 1
    if "q" in fen_info:
        tensor[13, 7, 0] = 1

    return tensor


class ChessDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fen = self.data.iloc[idx, 0]
        position = fen_to_tensor(fen)
        score = torch.tensor(
            np.float32(self.data.iloc[idx, 1]) / 100, dtype=torch.float32
        )
        return position, score


class ChessModel(nn.Module):
    def __init__(self):
        super(ChessModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(14 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        return self.layers(x)


dataset = ChessDataset("bigdata.csv")
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size]
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = ChessModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
writer = SummaryWriter()

sample_data = next(iter(train_loader))[0]
writer.add_graph(model, sample_data)

try:
    for epoch in range(30):
        running_loss = 0.0
        model.train()
        for i, batch in enumerate(train_loader):
            positions, scores = batch
            optimizer.zero_grad()
            outputs = model(positions)
            outputs = outputs.squeeze()
            loss = criterion(outputs, scores)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:
                avg_loss = running_loss / 10
                print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {avg_loss:.4f}")
                writer.add_scalar(
                    "training_loss", avg_loss, epoch * len(train_loader) + i
                )

                running_loss = 0.0

        # log histograms of model weights and gradients
        for name, param in model.named_parameters():
            writer.add_histogram(f"{name}_weights", param, epoch)
            writer.add_histogram(f"{name}_gradients", param.grad, epoch)

        model.eval()
        with torch.no_grad():
            validation_loss = 0.0
            for j, batch in enumerate(test_loader):
                positions, scores = batch
                outputs = model(positions)
                outputs = outputs.squeeze()
                test_loss = criterion(outputs, scores)
                validation_loss += test_loss.item()

        avg_validation_loss = validation_loss / len(test_loader)
        writer.add_scalar("validation_loss", avg_validation_loss, epoch)

finally:
    writer.close()
    torch.save(model.state_dict(), "models/model.pth")
