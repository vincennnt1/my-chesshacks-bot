import chess
import numpy as np

PIECE_TO_INDEX = {
    None: 0,
    chess.Piece(chess.PAWN, True): 1,
    chess.Piece(chess.KNIGHT, True): 2,
    chess.Piece(chess.BISHOP, True): 3,
    chess.Piece(chess.ROOK, True): 4,
    chess.Piece(chess.QUEEN, True): 5,
    chess.Piece(chess.KING, True): 6,
    chess.Piece(chess.PAWN, False): 7,
    chess.Piece(chess.KNIGHT, False): 8,
    chess.Piece(chess.BISHOP, False): 9,
    chess.Piece(chess.ROOK, False): 10,
    chess.Piece(chess.QUEEN, False): 11,
    chess.Piece(chess.KING, False): 12,
}

def board_to_tensor(board: chess.Board) -> np.ndarray:
    arr = np.zeros((64, 13), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        idx = PIECE_TO_INDEX[piece]
        arr[square, idx] = 1.0

    return arr.reshape(-1)  # returns NumPy array of shape (832,)
