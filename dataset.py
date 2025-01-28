import pandas as pd

import numpy as np

# import berserk
import chess
import chess.engine
import chess.pgn
import requests
import io
import tqdm
import time
import random
import json

import torch
from torch.utils.data import Dataset, DataLoader

DEBUG = True
# DEBUG = False


def log(text: str, *args, **kwargs):
    if DEBUG:
        print(f"[{time.strftime('%H:%M:%S')}] {text}", *args, **kwargs)


def GET(url: str):
    response = requests.get(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "QQQ-Position-Scraper",
            "X-Apology": "sorry for spamming your api :( i'll try to keep it to a minimum",
        },
    )
    log(f"GET {url} {response.status_code}")
    return response


def get_chesscom_titled_players(title: str):
    url = f"https://api.chess.com/pub/titled/{title.upper()}"
    response = GET(url)
    return response.json()["players"]


def get_game_archives(username: str):
    url = f"https://api.chess.com/pub/player/{username}/games/archives"
    response = GET(url)
    return response.json()["archives"]


def get_games_from_archive(url: str, num_games: int = 0):
    try:
        response = GET(url)
        return response.json()["games"], num_games + len(response.json()["games"])
    except KeyError:
        return [], num_games


def game_to_board(game: dict):
    board = chess.Board()
    try:
        pgn = game["pgn"]
    except KeyError:
        pgn = "1. e4"
    # log(f"game_to_board: {pgn}")
    game = chess.pgn.read_game(io.StringIO(pgn))
    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
    return board


def get_positions_from_board(board: chess.Board):
    for move in board.legal_moves:
        board.push(move)
        yield board.fen()
        board.pop()


def stockfish_evaluate_position(
    fen: str, engine: chess.engine.SimpleEngine, depth: int = 0
):
    try:
        board = chess.Board(fen)
    except ValueError:
        log(f"stockfish_evaluate_position: invalid fen: {fen}")
        return np.nan
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    # score = info["score"].white().score()
    if board.turn == chess.WHITE:
        score = info["score"].white().score(mate_score=100000)
    else:
        score = info["score"].black().score(mate_score=100000)
    # log(f"stockfish_evaluate_position: {fen} {score}")
    return score


def generate_dataset(
    # username: str,
    # title: str,
    mode: str = "title",
    username: str = "hikaru",
    title: str = "GM",
    user_limit: int = 1,
    game_limit_per_user: int = 20,
):
    start = time.time()
    if mode == "title":
        titled_players = get_chesscom_titled_players(title)
        random.shuffle(titled_players)
        game_archives = []
        for player in tqdm.tqdm(
            titled_players[:user_limit], desc="fetching game archives", unit="user"
        ):
            game_archives.extend(get_game_archives(player))
        random.shuffle(game_archives)
    elif mode == "user":
        game_archives = get_game_archives(username)
        random.shuffle(game_archives)
    elif mode != "cached":
        raise ValueError("mode must be either 'title', 'user', or 'cached'")

    if mode == "cached":
        log("generate_dataset: using cached games")
        with open("games.json", "r") as f:
            games = json.loads(f.read())
        num_games = len(games)
        log(f"generate_dataset: loaded {num_games} games from cache")
    else:
        num_games = 0
        game_limit = game_limit_per_user * user_limit
        games = []
        for archive in tqdm.tqdm(
            game_archives,
            desc="fetching games",
            unit="archive",
        ):
            new_games, num_games = get_games_from_archive(archive, num_games=num_games)
            games.extend(new_games)
            if len(games) >= game_limit:
                break

        with open("games.json", "w+") as f:
            f.write(json.dumps(games))

    log(f"generate_dataset: sample game: {games[0]}")

    boards = []
    for game in tqdm.tqdm(
        games,
        desc="converting games to boards",
        unit="game",
    ):
        boards.append(game_to_board(game))
        games.remove(game)  # memory management

    positions = set()  # save a bit of time and memory
    for board in tqdm.tqdm(boards, desc="getting positions from boards", unit="board"):
        positions.update(get_positions_from_board(board))

    positions = list(positions)

    engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    scores = []
    for position in tqdm.tqdm(positions, desc="evaluating positions", unit="position"):
        score = stockfish_evaluate_position(position, engine)
        if score != np.nan:
            scores.append(score)
        else:
            positions.remove(position)
    engine.quit()
    end = time.time()

    df = pd.DataFrame({"position": positions, "score": scores})

    df.to_csv(f"dataset_{mode}_{username}_{title}.csv", index=False)

    log(
        f"generate_dataset: analyzed {len(positions)} positions in {end - start} seconds"
    )

    return df


# def square_to_index(square:str):
#     file_letter, rank_number = square[0], int(square[1])
#     column = ord(file_letter) - ord('a')
#     row = 8 - rank_number

#     return row, column

# def split_dims(fen: str):
#     board = chess.Board(fen)

#     board3d = np.zeros((18, 8, 8), dtype=np.int8)

#     castling_rights_mapping = {
#         'K': 12, 'Q': 13, 'k': 14, 'q': 15
#     }

#     for piece in chess.PIECE_TYPES:
#         for square in board.pieces(piece, chess.WHITE):
#             idx = np.unravel_index(square, (8, 8))
#             board3d[piece - 1][7 - idx[0]][idx[1]] = 1
#         for square in board.pieces(piece, chess.BLACK):
#             idx = np.unravel_index(square, (8, 8))
#             board3d[piece + 5][7 - idx[0]][idx[1]] = 1

#     for color in [chess.WHITE, chess.BLACK]:
#         for rights, index in castling_rights_mapping.items():
#             if board.has_kingside_castling_rights(color, rights.lower()):
#                 board3d[index] = 1
#             if board.has_queenside_castling_rights(color, rights.lower()):
#                 board3d[index + 1] = 1

#     board3d[16] = int(board.turn)

#     ep_square = board.ep_square # holy hell
#     if ep_square is not None:
#         idx = np.unravel_index(ep_square, (8, 8))
#         board3d[17][7 - idx[0]][idx[1]] = 1

#     return board3d

# very ugly but works
# thx https://medium.com/@nihalpuram/training-a-chess-ai-using-tensorflow-e795e1377af2

squares_index = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}


def square_to_index(square):
    letter = chess.square_name(square)
    return 8 - int(letter[1]), squares_index[letter[0]]


def split_dims(board):
    board3d = np.zeros((14, 8, 8), dtype=np.int8)
    for piece in chess.PIECE_TYPES:
        for square in board.pieces(piece, chess.WHITE):
            idx = np.unravel_index(square, (8, 8))
            board3d[piece - 1][7 - idx[0]][idx[1]] = 1
        for square in board.pieces(piece, chess.BLACK):
            idx = np.unravel_index(square, (8, 8))
            board3d[piece + 5][7 - idx[0]][idx[1]] = 1
        aux = board.turn
        board.turn = chess.WHITE
        for move in board.legal_moves:
            i, j = square_to_index(move.to_square)
            board3d[12][i][j] = 1
        board.turn = chess.BLACK
        for move in board.legal_moves:
            i, j = square_to_index(move.to_square)
            board3d[13][i][j] = 1
    board.turn = aux
    return board3d


class ChessDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fen = self.data.iloc[idx, 0]
        score = self.data.iloc[idx, 1]
        board = chess.Board(fen)
        x = split_dims(board)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(score, dtype=torch.float32)
        if self.transform:
            x = self.transform(x)
        return x, y


if __name__ == "__main__":
    generate_dataset(
        mode="title",
        # username="hikaru",
        user_limit=125,
        game_limit_per_user=np.inf,
    )
    # generate_dataset(
    #     mode="user",
    #     username="hikaru",
    #     # user_limit=10,
    #     game_limit_per_user=10,
    # )
    # test analyze 20 hikaru games
    exit(0)
