"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    if terminal(board):
        return 0

    turns = 0

    for i in range(len(board)):  # Verificar os movimentos feitos
        for j in range(len(board[i])):
            if board[i][j] != EMPTY:
                turns += 1

    # Se o numero de movimentos for par, então é o X a jogar se não é o O
    if turns % 2 == 0:
        return X
    else:
        return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """

    if terminal(board):
        return 0
    actions = set()
    for i in range(len(board)):  # Apenas é possível jogar se não for EMPTY
        for j in range(len(board[i])):
            if board[i][j] == EMPTY:
                action = (i, j)
                actions.add(action)
    return actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if board[action[0]][action[1]] != EMPTY:  # Exceção se não estiver empty
        raise Exception

    if (
        action[0] < 0 or action[0] > 2 or action[1] > 2 or action[1] < 0
    ):  # Exceção se for inválido
        raise Exception
    temp = copy.deepcopy(board)  # criar a cópia e fazer o movimento
    temp[action[0]][action[1]] = player(board)

    return temp


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # Verificar todas as formas de ganhar um jogo do galo
    diagonal1 = board[0][0] == board[1][1] == board[2][2] != EMPTY
    diagonal2 = board[0][2] == board[1][1] == board[2][0] != EMPTY
    if diagonal1 or diagonal2:
        return board[1][1]
    for i in range(len(board)):
        horizontal = board[i][0] == board[i][1] == board[i][2] != EMPTY
        if horizontal:
            return board[i][0]
        vertical = board[0][i] == board[1][i] == board[2][i] != EMPTY
        if vertical:
            return board[0][i]
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """

    if winner(board) != None:  # Se houver um vencedor, então é game over
        return True

    for i in range(len(board)):  # Se não estiver cheiro ainda está a decorrer
        for j in range(len(board[i])):
            if board[i][j] == EMPTY:
                return False
    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """

    if winner(board) == O:
        return -1
    elif winner(board) == X:
        return 1
    else:
        return 0


def MaxValue(state):
    """
    Returns the biggest value action its value (action, value)
    """
    if terminal(state):
        return (None, utility(state))

    v = -9999999

    for action in actions(state):
        value = MinValue(result(state, action))[1]
        if v < value:
            v = value
            finalAction = action

    return (finalAction, v)


def MinValue(state):
    """
    Returns the smallest value action its value (action, value)
    """
    if terminal(state):
        return (None, utility(state))

    v = 9999999

    for action in actions(state):
        value = MaxValue(result(state, action))[1]
        if v > value:
            v = value
            finalAction = action

    return (finalAction, v)


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if player(board) == X:
        return MaxValue(board)[0]
    elif player(board) == O:
        return MinValue(board)[0]
    else:
        return None
