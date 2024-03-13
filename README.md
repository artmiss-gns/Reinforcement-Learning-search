# Snake and Ladders Path-Finding

This project is an implementation of the classic 'Snake and Ladders' game, where the objective is to find the best path from the starting position (state 1) to the final position (state 100) while navigating through snakes and ladders on the board.

## Approach

The project explores two different approaches to solve the problem:

1. **Reinforcement Learning**: Using Q-learning, an agent learns the optimal action to take in each state by exploring the environment and updating a Q-table based on the rewards received.

2. **Informed Search Algorithm**: The A* algorithm is employed to find the shortest path from the start state to the goal state, using a heuristic function to guide the search.


## Getting Started

To run this project, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies by running:
```bash
pip install -r requirements.txt
```
3. Navigate to the desired directory (`src/A_star` or `src/RL/q_learning`).
4. Run the `main.py` script or open the corresponding Jupyter Notebook.

## Usage

1. **Reinforcement Learning (Q-learning)**:
   - Run the `main.py` script or the `main.ipynb` Jupyter Notebook in the `src/RL/q_learning` directory.
   - The agent will learn the optimal policy by exploring the environment and updating the Q-table.
   - You can visualize the learning process and the final policy.
   - The learned Q-table will be saved as a pickle file (`answer_qtable.pickle` and `answer_qtable(epsilon-greedy).pickle`) for future use.

2. **A\* Algorithm**:
   - Run the `main.py` script or the `main.ipynb` Jupyter Notebook in the `src/A_star` directory.
   - The A* algorithm will find the shortest path from the start state to the goal state using the provided heuristic function.
   - You can visualize the explored states and the final path.


## Some Notes
- **Best Model's Q-table**: The Q-table learned by the best Q-learning model is saved as a pickle file (`answer_qtable.pickle` and `answer_qtable(epsilon-greedy).pickle`) for future use.
- **Error Handling**: Errors are handled and logged using the `logging` library, with log files stored in the `logs` directory.