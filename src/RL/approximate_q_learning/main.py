from src.RL.approximate_q_learning.approximate_Qlearning import Approximate_Qlearning

snakes = [(42,96), (71,94), (32,75), (16,47), (10,25), (3,37)]
ladders = [(54,88), (41,79), (22,58), (14,55), (12,50), (4,56)]

n = 10

game = Approximate_Qlearning(n=n, snakes=snakes, ladders=ladders, alpha=0.5, p=0.7, lambda_=1, log=False)
game.train()