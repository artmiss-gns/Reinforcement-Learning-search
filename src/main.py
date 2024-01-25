from typing import NamedTuple, List
import numpy as np
import pandas as pd

from RL.QLearning import QLearning
from src.utils.utils import show_optimal_policy, show_table

snakes = [(42,96), (71,94), (32,75), (16,47), (10,25), (3,37)]
ladders = [(54,88), (41,79), (22,58), (14,55), (12,50), (4,56)]

n = 10
game = QLearning(n=n, snakes=snakes, ladders=ladders, log=True, alpha=0.5, p=0.7, epsilon_greedy=True)
game.run(n_iter=8000)
# create_q_table_frame(game)

show_table(game.n)
print(f"\nsnakes: \n", game.snakes, end="\n\n")
print(f"ladders: \n", game.ladders)

print(show_optimal_policy(game))
game.show_path()