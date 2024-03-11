import numpy as np
import pandas as pd
from src.RL.q_learning.action import Actions

def show_table(n: int) :
    board = np.arange((n**2), 0, step=-1).reshape(n, n)
    # reverse the odd rows
    if n%2 == 0 :
        odd_rows = np.apply_along_axis(
            func1d=(lambda r: r[::-1]),
            axis=1,
            arr=board[1::2, ]
        )
        board[1::2,] = odd_rows

    else :
        even_rows = np.apply_along_axis(
            func1d=(lambda r: r[::-1]),
            axis=1,
            arr=board[0::2, ]
        )
        board[0::2,] = even_rows

    print(board)


def create_q_table_frame(game) :
    q_table = game.q_table.copy()
    q_table_frame = pd.DataFrame(
        q_table,
        columns=Actions._actions,
        index=np.arange(1, (game.n**2)+1)
    )
    for row in q_table_frame.iterrows() :
        actions = game.get_actions(game.state_to_loc(row[0], game.n))
        nan_actions = q_table_frame.loc[row[0], :].drop(actions).index
        q_table_frame.loc[row[0], nan_actions] = np.nan
        # print(actions)
    return q_table_frame

def show_q_table(game) :
    print(create_q_table_frame(game))

def show_optimal_policy(game) :
    q_table_frame = create_q_table_frame(game)
    return q_table_frame.idxmax(axis=1)


        