from .utils import chess_manager, GameContext
import chess
from chess import Move
import random
import time

from .model import TinyChessModel
from .encoder import board_to_tensor

# Write code here that runs once
# Can do things like load models from huggingface, make connections to subprocesses, etcwenis

# -------------------------
# Load the trained model
# -------------------------
model = None
def get_model():
    global model
    if model is None:
        print("Loading model...")
        model = TinyChessModel("src/utils/model_weights/model_weights_distilled.npz")
    return model

# ================================
# Classical Evaluation
# ================================
PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:     0,
}

# global things
eval_cache = {}
node_count = 0
MAX_NODES = 24000
MAX_Q_DEPTH = 4


def classical_eval(board: chess.Board):
    score = 0

    # Material
    for piece_type, val in PIECE_VALUES.items():
        score += len(board.pieces(piece_type, chess.WHITE)) * val
        score -= len(board.pieces(piece_type, chess.BLACK)) * val

    # Simple king safety
    if board.fullmove_number > 8:
        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)

        # penalize exposed kings in center files (d/e)
        if chess.square_file(white_king) in [3, 4]:  # d or e file
            score -= 40
        if chess.square_file(black_king) in [3, 4]:
            score += 40

        # small bonus for castling
        if board.has_castling_rights(chess.WHITE):
            pass
        else:
            score += 15  # white has castled or lost the right

        if board.has_castling_rights(chess.BLACK):
            pass
        else:
            score -= 15
    
    # ===========================
    # MOBILITY BONUS
    # ===========================

    # Detect opening: both queens still on board + many minor pieces
    opening_phase = (
        len(board.pieces(chess.QUEEN, True)) + len(board.pieces(chess.QUEEN, False)) == 2
        and len(board.pieces(chess.KNIGHT, True)) + len(board.pieces(chess.KNIGHT, False)) >= 3
        and len(board.pieces(chess.BISHOP, True)) + len(board.pieces(chess.BISHOP, False)) >= 3
    )


    # Phase-dependent weights
    if opening_phase:
        KNIGHT_MOB = 3
        BISHOP_MOB = 3
        ROOK_MOB   = 1
    else:
        KNIGHT_MOB = 2
        BISHOP_MOB = 2
        ROOK_MOB   = 3

    def mobility_for(color):
        bonus = 0

        for move in board.legal_moves:
            # only count moves for this color
            piece = board.piece_at(move.from_square)
            if piece is None or piece.color != color:
                continue

            piece_type = piece.piece_type
            to_sq = move.to_square

            # skip unsafe moves
            if board.is_attacked_by(not color, to_sq):
                continue

            # QUEEN STUFF
            if (opening_phase and piece_type == chess.QUEEN):
                bonus -= 40

            # forward mobility bonus
            forward_bonus = 0
            if opening_phase:
                if color == chess.WHITE:
                    if chess.square_rank(to_sq) > chess.square_rank(move.from_square):
                        forward_bonus = 1
                else:
                    if chess.square_rank(to_sq) < chess.square_rank(move.from_square):
                        forward_bonus = 1

            # apply piece-specific weights
            if piece_type == chess.KNIGHT:
                bonus += KNIGHT_MOB + forward_bonus
            elif piece_type == chess.BISHOP:
                bonus += BISHOP_MOB + forward_bonus
            elif piece_type == chess.ROOK:
                bonus += ROOK_MOB

        return bonus

    score += mobility_for(chess.WHITE)
    score -= mobility_for(chess.BLACK)

    return score


# ================================
# Hybrid NN + Classical Evaluation
# ================================
def nn_eval(board):
    m = get_model()
    x = board_to_tensor(board)
    return m.forward(x)


# NO TIME TO FIX ALL CODE, but essentially ONLY uses NN now, no classical eval heuristics
def evaluate(board, use_nn):
    # --- caching key ---
    key = board.fen()

    # classical-only positions can also be cached!
    cache_key = (key, use_nn)

    if cache_key in eval_cache:
        return eval_cache[cache_key]

    # compute score
    classical = classical_eval(board)

    if use_nn:
        neural = nn_eval(board)
        score = classical * 0.25 + neural * 0.75
    else:
        score = classical
        neural = nn_eval(board)

    eval_cache[cache_key] = score
    return neural


# ================================
# Move Ordering (captures & checks first)
# ================================
def move_score(board, move):
    piece = board.piece_at(move.from_square)

    score = 0
    
    # Favor captures/checks
    if board.is_capture(move):
        score += 1000
    if board.gives_check(move):
        score += 500

    # Massive queen penalty in opening
    if piece and piece.piece_type == chess.QUEEN:
        # File: early game = no queen adventures
        if board.fullmove_number <= 10:
            score -= 800  # enough to override capture bias

    return score


# ================================
# Quiescence Search
# ================================
def quiescence(board, alpha, beta, use_nn, q_depth=0):
    global node_count
    node_count += 1
    
    if node_count > MAX_NODES:
        return evaluate(board, use_nn=False)
    
    if q_depth >= MAX_Q_DEPTH:
        return evaluate(board, use_nn=False)
     
    stand_pat = evaluate(board, use_nn=(use_nn and q_depth == 0))

    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    for move in board.legal_moves:
        if not board.is_capture(move):
            continue

        board.push(move)
        score = -quiescence(board, -beta, -alpha, use_nn, q_depth=q_depth + 1)
        board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha


# ================================
# Alpha-Beta Minimax Search
# ================================
def search(board, depth, alpha, beta):
    global node_count
    node_count += 1

    if node_count > MAX_NODES:
        return evaluate(board, use_nn=False)

    if depth == 0:
        return quiescence(board, alpha, beta, use_nn=True, q_depth=0)

    best_value = -999999

    moves = list(board.legal_moves)
    moves.sort(key=lambda m: move_score(board, m), reverse=True)

    for move in moves:
        board.push(move)
        value = -search(board, depth-1, -beta, -alpha)
        board.pop()

        if value > best_value:
            best_value = value
        if best_value > alpha:
            alpha = best_value
        if alpha >= beta:
            break

    return best_value


# ================================
# Picking the Best Move
# ================================
def choose_best_move(board, depth=3):
    global node_count
    node_count = 0

    legal_moves = list(board.legal_moves)

    best_score = -999999
    best_move = random.choice(legal_moves)

    # order root moves
    legal_moves.sort(key=lambda m: move_score(board, m), reverse=True)

    for move in legal_moves:
        board.push(move)
        score = -search(board, depth - 1, -999999, 999999)
        board.pop()

        if score > best_score:
            best_score = score
            best_move = move
    
    return best_move


# ================================
# Integration with chess_manager
# ================================
@chess_manager.entrypoint
def test_func(ctx: GameContext):
    try:
        print("Engine thinking with search + NN...")
        time.sleep(0.05)

        best_move = choose_best_move(ctx.board, depth=2)
        return best_move
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e

@chess_manager.reset
def reset_func(ctx: GameContext):
    global eval_cache
    eval_cache = {}
    pass
