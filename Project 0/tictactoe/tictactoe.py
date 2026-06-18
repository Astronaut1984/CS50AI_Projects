"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None
GRID_SIZE = 3   # We are a no magic numbers household

# The rows, columns and diagonals for checking win conditions
ROWS = [
    [(0, 0), (0, 1), (0, 2)],
    [(1, 0), (1, 1), (1, 2)],
    [(2, 0), (2, 1), (2, 2)],
]

COLS = [
    [(0, 0), (1, 0), (2, 0)],
    [(0, 1), (1, 1), (2, 1)],
    [(0, 2), (1, 2), (2, 2)],
]

DIAGS = [
    [(0, 0), (1, 1), (2, 2)],
    [(0, 2), (1, 1), (2, 0)],
]

LINES = ROWS + COLS + DIAGS


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """

    # calculate the number of Xs and Os, then return X
    # if there are more O's or an equal number, O otherwise

    numX = 0
    numO = 0
    for row in board:
        for cell in row:
            if cell == X:
                numX += 1
            elif cell == O:
                numO += 1
    
    if numX <= numO:
        return X
    else:
        return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    # loop all over the board and find empty cells to fill
    moves = set()
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if board[i][j] == EMPTY:
                moves.add((i, j))

    return moves


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    p = player(board)
    i, j = action
    if i >= GRID_SIZE or j >= GRID_SIZE or i < 0 or j < 0 or board[i][j] is not EMPTY:
        raise KeyError("Invalid action")
    board_copy = copy.deepcopy(board)   # Thanks for the copy library reference docs cs50, very nice
    board_copy[i][j] = p
    return board_copy


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    for line in LINES:
        cells = [board[i][j] for i, j in line]
        # thanks to https://www.geeksforgeeks.org/python/python-check-if-all-elements-in-a-list-are-same/
        if cells[0] is not EMPTY and len(set(cells)) == 1:
            return cells[0]
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    # The game is over if one player wins or a tie happens, so I will check for both

    if winner(board) is not None:
        return True

    for row in board:
        for cell in row:
            if cell is EMPTY:
                return False
            
    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    win_player = winner(board)
    if win_player == X:
        return 1
    elif win_player == O:
        return -1
    else:
        return 0


# I separated to min_value and max_value so I can follow lecture pseudocode
def min_value(board):
    if terminal(board):
        return utility(board), None
    
    assert len(actions(board)) > 0, f"No actions but board is not terminal:\n{board}"

    v = math.inf
    best_action = None
    
    for action in actions(board):
        score, _ = max_value(result(board, action))
        if score < v:
            v = score
            best_action = action
    return v, best_action


def max_value(board):
    if terminal(board):
        return utility(board), None
    
    assert len(actions(board)) > 0, f"No actions but board is not terminal:\n{board}"

    v = -math.inf
    best_action = None
    
    for action in actions(board):
        score, _ = min_value(result(board, action))
        if score > v:
            v = score
            best_action = action

    return v, best_action


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None
    
    p = player(board)
    if p == X:
        _, res = max_value(board)
        return res
    else:
        _, res = min_value(board)
        return res