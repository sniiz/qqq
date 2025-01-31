import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.utils.tensorboard.writer import SummaryWriter  # Import TensorBoard
import pandas as pd
import numpy as np
import tqdm
import chess
import random
import time

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


def algebraic_to_ucimove(algebraic_move, board):
    for move in board.legal_moves:
        if board.san(move) == algebraic_move:
            return move.uci()
    return None


model = ChessModel()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("models/model_but_good.pth", map_location=device))
model.to(device)


def count_material(fen):
    board = chess.Board(fen)
    piece_values = {
        "P": 1,
        "N": 3,
        "B": 3,
        "R": 5,
        "Q": 9,
        "p": -1,
        "n": -3,
        "b": -3,
        "r": -5,
        "q": -9,
        "K": 0,
        "k": 0,
    }
    material = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            material += piece_values[piece.symbol()]
    return material


def evaluate(fen, history=[]):
    board = chess.Board(fen)
    if board.is_game_over():
        result = board.result()
        if (
            result == "1-0" and board.turn == chess.WHITE
        ):  # incredible programming over here
            return float("inf")
        if result == "0-1" and board.turn == chess.BLACK:
            return float("inf")
        if result == "1-0" and board.turn == chess.BLACK:
            return float("-inf")
        if result == "0-1" and board.turn == chess.WHITE:
            return float("-inf")
        return 0
    model.eval()
    with torch.no_grad():
        tensor = fen_to_tensor(fen).unsqueeze(0)  # Add batch dimension
        tensor = tensor.to(device)
        output = model(tensor)
        # output = round(output.item(), 3)
        output = output.item()

        material = count_material(fen)
        output += material / 2

        return output


def minimax(
    board, max_depth, current_depth=0, is_maximizing=False, move_history=[]
):  # hypothetically we have no reason to minimize since the evaluation function returns the score from its perspective
    if current_depth == max_depth or board.is_game_over():
        return evaluate(board.fen(), move_history), None

    best_score = float("-inf") if is_maximizing else float("inf")
    best_move = None
    # for move in board.legal_moves:
    legal_moves = list(board.legal_moves)
    random.shuffle(legal_moves)

    if current_depth == 0:
        legal_moves = tqdm.tqdm(legal_moves, unit="branches", unit_scale=True)

    for move in legal_moves:
        board.push(move)
        score, _ = minimax(
            board,
            max_depth,
            current_depth + 1,
            not is_maximizing,
            move_history + [move],
        )
        board.pop()
        if is_maximizing:
            if score > best_score:
                if current_depth == 0:
                    print(f"new best: {board.san(move)} : {round(score, 2)}")
                best_score = max(best_score, score)
                best_move = move
        else:
            if score < best_score:
                if current_depth == 0:
                    print(f"new best: {board.san(move)} : {round(score, 2)}")
                best_score = min(best_score, score)
                best_move = move

    return best_score, best_move


# Get best move using minimax
def best_move(fen, depth=3):
    board = chess.Board(fen)
    _, best_move = minimax(board, depth)
    return best_move


# CLI Game Loop
def play_chess():
    board = chess.Board()

    while not board.is_game_over():
        print()
        print(board)  # ASCII representation of the board

        if board.turn == chess.WHITE:
            move = input("\nyour move (in algebraic notation): ")
            if move in ["quit", "exit"]:
                print("goodbye!")
                break
            try:
                board.push_san(move)
            except ValueError:
                print("invalid move. try again.")
                continue
        else:
            print("\n[AI TURN] ai is thinking...\n")
            ai_move = best_move(board.fen(), depth=3)
            if ai_move:
                print(f"\n[AI MOVE] ai plays: {board.san(ai_move)}")
                board.push(ai_move)

            else:
                print("ai resigns.")
                break

    print("\n[GAME OVER] final position:")
    print(board)
    print(f"result: {board.result()}")


# Start the game
if __name__ == "__main__":
    play_chess()
