import torch
from torch import nn
import numpy as np


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


def fen_to_tensor(fen):
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


def evaluate_fen(fen, model):
    model.eval()
    with torch.no_grad():
        tensor = fen_to_tensor(fen).unsqueeze(0)  # Add batch dimension
        output = model(tensor)
        return output.item()


model = ChessModel()
model.load_state_dict(torch.load("models/model.pth"))

fen_input = input("Enter FEN string: ")
evaluation = evaluate_fen(fen_input, model)
print(f"Model evaluation: {evaluation:.4f}")
