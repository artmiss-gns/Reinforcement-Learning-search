from typing import NamedTuple, List
import numpy as np
import pandas as pd

from QLearning import QLearning
from src.utils.utils import show_q_table, show_table

snakes = [(1, 7)]
ladders = [(2, 8)]

game = QLearning(n=3, snakes=snakes, ladders=ladders)
game.run()
show_q_table(game)

print("\n\n")
show_table(game.n)
print("\n\n")
print(f"snakes: \n", game.snakes, end="\n\n")
print(f"ladders: \n", game.ladders)

print(game.loc_to_state((0, 2), game.n))
print(game.get_actions((0, 2)))