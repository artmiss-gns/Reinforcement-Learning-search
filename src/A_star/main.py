from src.A_star import A_star

snakes = [(42,96), (71,94), (32,75), (16,47), (10,25), (3,37)]
ladders = [(54,88), (41,79), (22,58), (14,55), (12,50), (4,56)]


n = 10
game = A_star(start_state=1, n=10, snakes=snakes, ladders=ladders, )
game.run()