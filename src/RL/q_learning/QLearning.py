from typing import List, Tuple
import numpy as np
import logging

from src.RL.q_learning.action import Actions

class QLearning :
    def __init__(
            self, n, snakes:List[Tuple[int, int]], ladders:List[Tuple[int, int]],
            alpha=0.3, p=0.7, lambda_=1, log=True, epsilon_greedy=False
        ) :
        self.n = n
        self.snakes = snakes
        self.ladders = ladders
        self.snakes_loc = []
        self.ladders_loc = []
        self.alpha = alpha
        self.p = p
        self.lambda_ = lambda_
        self.q_table = np.zeros(shape=(self.n**2, Actions.get_num_actions()))
        self.log = log
        self.epsilon_greedy = epsilon_greedy
        self.epsilon_discount_factor = 0.5 # a value which is going to be reduced from epsilon

        self.complete_cycle = 0

        # fix snakes loc
        if snakes :
            for s in snakes :
                x_start, y_start = self.state_to_loc(s[0], self.n)
                x_end, y_end = self.state_to_loc(s[1], self.n)
                self.snakes_loc.append(
                    ((x_start, y_start, x_end, y_end))
                )
        else :
            self.snakes_loc = []
            
        # fix ladder loc
        if ladders :
            for l in ladders :
                x_start, y_start = self.state_to_loc(l[0], self.n)
                x_end, y_end = self.state_to_loc(l[1], self.n)
                self.ladders_loc.append(
                    ((x_start, y_start, x_end, y_end))
                )
        else :
            self.ladders_loc = []

        # * (x, y, x', y') -> (x, y) for start / (x', y') for the end
        self.snakes_loc = np.array(self.snakes_loc)
        self.ladders_loc = np.array(self.ladders_loc)

        # make q-table
        self._create_q_table()

        # configure logging
        if log : 
            logging.basicConfig(filename='../../logs/log.txt', level=logging.DEBUG, format='%(message)s')
            self.logger = logging.getLogger()

    def _create_q_table(self) :
        for state in range(1, (self.n**2)+1):
            loc = self.state_to_loc(state, self.n)
            actions = self.get_actions(loc)
            nan_actions = [a for a in Actions._actions if a not in actions]
            nan_actions_index = [Actions.get_index(a) for a in nan_actions]
            self.q_table[state-1, [nan_actions_index]] = np.NaN



            
    def inbound(self, loc, x=0, y=0) :
        new_loc = (loc[0]+y, loc[1]+x)
        if (0 <= new_loc[0] < self.n) and (0 <= new_loc[1] < self.n) :
            return True
        return False

    def get_actions(self, loc) -> List[Actions]:
        # TODO this function can be changed , there is no need to check all the action from scratch, it can be read from q-table
        if self.loc_to_state(loc, self.n) == self.n**2 : # final state
            return [Actions.terminate]
        elif self.snakes and list(loc) in self.snakes_loc[:, -2:].tolist(): # if we are at the head of a snake
            return [Actions.snake_down]
        elif self.ladders and list(loc) in self.ladders_loc[:, 0:2].tolist(): # if we are at the bottom of a ladder
            return [Actions.ladder_up]
        else :
            # ! if we could NOT move up in all the side states, uncomment this part :
            actions = [Actions.right_1, Actions.right_2, Actions.left_1, Actions.left_2,]
            if (loc[0]%2 == 0) and (loc[1] == (self.n-1)) : # even rows
                actions.extend([Actions.up]) #  [..,Actions.down] if we could come down from sides
            elif (loc[0]%2 != 0) and (loc[1] == (0)) : # odd rows
                actions.extend([Actions.up]) #  [..,Actions.down] if we could come down from sides

            # ! if we could move up in all the side states, uncomment this part :
            # actions = [Actions.right_1, Actions.right_2, Actions.left_1, Actions.left_2,]
            # if (loc[1] == (self.n-1)) or (loc[1] == (0))  : # even rows
            #     actions.extend([Actions.up]) #  [..,Actions.down] if we could come down from sides
            
                
            return actions
            # ? we also could have improved this by omitting the actions that are going to go out of the bounds of board
            # actions=[]
            # # 1 right
            # if inbound(loc, x=1) :
            #     actions.append(Actions.right_1)
            # elif inbound(loc, x=2) :

    @staticmethod
    def loc_to_state(loc, n) -> int:
        '''Converts the (x,y) points to the state number of the actual game board
        loc = (x, y)
            x is even  --> (x*n) + (y+1)
            x is odd --> (x*n) + (n-y)
        '''
        x, y = loc
        if x%2 == 0 :
            return (x*n) + (y+1)
        if x%2 != 0: 
            return (x*n) + (n-y)
        
    @staticmethod 
    def state_to_loc(state, n) -> List[Tuple[int, int]] :
        '''
        we had these formulas for `loc_to_state` function, now by having x, we can 
        have y from the formula :
            x is even  --> (x*n) + (y+1) = state
            x is odd --> (x*n) + (n-y) = state
        '''
        x = (state - 1) // n
        # even row (starting from 0)
        if  x%2 == 0 :
            x = (state - 1) // n
            y = (state - (x*n)) - 1

        # odd row
        else :
            x = (state - 1) // n
            y = -1*((state - (x*n))-n)



        return (x, y)

    def bellman(self, loc, new_loc, action) :
        '''
        sample = R(s,a,s') + λ*max Q(s',a')
        Q(s,a) = (1-a)*Q(s,a) + a*sample 
        '''
        if new_loc == -1 : # we are in the final step # * we can make this part better later...
            value = Actions.get_reward(action) # ? does it need Lambda to be multiplied to it 
        else :
            state = self.loc_to_state(loc, self.n)
            new_state = self.loc_to_state(new_loc, self.n)
            sample = Actions.get_reward(action) + self.lambda_*np.nanmax(self.q_table[new_state-1, :])   
                                                                                                # ! USE np.nanmax here, otherwise it
                                                                                                # ! will return Nan as max which is messed up!
            value = (1 - self.alpha) * self.q_table[state-1, Actions.get_index(action)] + self.alpha*sample
              
            
        return np.round(value, 3)
    
    def move(self, loc, action) : # -> tuple|int:
        '''
        by receiving a loc and action, it will return the new location
        if the return value is -1 the program should terminate
        '''
        if action == Actions.right_1 :
            if self.inbound(loc, x=1) :
                new_loc = (loc[0], loc[1]+1)
            else :
                new_loc = loc

        elif action == Actions.right_2 :
            if self.inbound(loc, x=2) :
                new_loc = (loc[0], loc[1]+2)
            else :
                if self.inbound(loc, x=1) :
                    new_loc = (loc[0], loc[1]+1)
                else :
                    new_loc = loc

        elif action == Actions.left_1 :
            if self.inbound(loc, x=-1):
                new_loc = (loc[0], loc[1]-1)
            else :
                new_loc = loc

        elif action == Actions.left_2 :
            if self.inbound(loc, x=-2):
                new_loc = (loc[0], loc[1]-2)
            else :
                if self.inbound(loc, x=-1) :
                    new_loc = (loc[0], loc[1]-1)
                else :
                    new_loc = loc

        elif action == Actions.up :
            if self.inbound(loc, y=1):
                new_loc = (loc[0]+1, loc[1])
            else : # ? this case should not happen logically 🤔
                new_loc = loc

        elif action == Actions.down :
            if self.inbound(loc, y=-1):
                new_loc = (loc[0]-1, loc[1])
            else :
                new_loc = loc

        elif action == Actions.ladder_up :
            # we are currently at the bottom of the ladder, so we get it's top :
            current_state = self.loc_to_state(loc, self.n)
            for bottom, top  in self.ladders :
                if bottom == current_state :
                    new_loc = self.state_to_loc(top, self.n)
                    break

        elif action == Actions.snake_down :
            current_state = self.loc_to_state(loc, self.n)
            for bottom, top  in self.snakes :
                if top == current_state :
                    new_loc = self.state_to_loc(bottom, self.n)
                    break

        elif action == Actions.terminate :
            return -1

        else :
            raise Exception(f"action` {action} ` not found!")
        return new_loc

    @staticmethod
    def minmax_scale(x: np.ndarray) -> np.ndarray:
        return (x - np.min(x)) / (np.max(x) - np.min(x))

    def perturb_action(self, loc, action, actions, epsilon) :
        if self.epsilon_greedy and len(actions) > 1: # we must have at least 2 option for actions
            state = self.loc_to_state(loc, self.n)
            if np.random.rand() < (epsilon-self.epsilon_discount_factor) : # randomness  / the epsilon itself seems to be too large
                indexes = [Actions.get_index(a) for a in actions]
                values = self.q_table.copy()[state-1, indexes].tolist()
                # removing the best action, since we want to select from other actions / # ! can be commented


                for ind, v in enumerate(values) :   # getting the index of the max value
                    if v == max(values) :
                        max_index = ind
                        break

                indexes.pop(max_index)
                values.pop(max_index) 
                values = np.array(values) # convert back values to an array
                actions.pop(max_index)

                weights = np.exp(values)/np.nansum(np.exp(values))
                weights = np.nan_to_num(weights, nan= 0) 
                weights = weights / np.sum(weights) # normalize the weights

                chosen_value = np.random.choice(values, p=weights)
                chosen_index = np.where(values == chosen_value)[0][0]

                action = actions[chosen_index]
            else :
                pass

        if action == Actions.up: action = np.random.choice([Actions.up, Actions.down], p=[self.p, 1-self.p])
        elif action == Actions.right_1: action = np.random.choice([Actions.right_1, Actions.left_1], p=[self.p, 1-self.p])
        elif action == Actions.right_2: action = np.random.choice([Actions.right_2, Actions.left_2], p=[self.p, 1-self.p])
        elif action == Actions.left_1: action = np.random.choice([Actions.left_1, Actions.right_1], p=[self.p, 1-self.p])
        elif action == Actions.left_2: action = np.random.choice([Actions.left_2, Actions.right_2], p=[self.p, 1-self.p])
        elif action == Actions.down: action = np.random.choice([Actions.down, Actions.up], p=[self.p, 1-self.p])
        elif action == Actions.ladder_up: action = Actions.ladder_up
        elif action == Actions.snake_down: action = Actions.snake_down
        elif action == Actions.terminate: action = Actions.terminate

        return action
    
    def get_best_move(self, loc, actions) :
        state = self.loc_to_state(loc, self.n)
        if actions[0] == Actions.ladder_up :
            # best_value = self.q_table[state-1, Actions.get_index(Actions.ladder_up)]
            best_action = Actions.ladder_up # which is also the only action
            return best_action
        
        elif actions[0] == Actions.snake_down :
            # best_value = self.q_table[state-1, Actions.get_index(Actions.snake_down)]
            best_action = Actions.snake_down # which is also the only action
            return best_action

        action_indexes = [Actions.get_index(a) for a in actions]
        if ((self.q_table[state-1, action_indexes]) == 0).all() : # when all the q-values have the same value, we choose one randomly
            # best_value = 0
            best_action = np.random.choice(actions)
        else :
            # best_value = max(self.q_table[state-1, action_indexes]) 
            best_action_index = np.argmax(self.q_table[state-1, action_indexes])
            best_action = actions[best_action_index] # ! DONT DO THIS  `Actions.index_to_action(best_action_index)` it returns a complete wrong
                                                    # ! wrong answer. the reason is we change the indexing area by `action_indexes` in 
                                                    # ! `q_table[state-1, action_indexes]`

        return best_action #, best_value

    def update_value(self, q_table:np.array, loc:tuple, action:Actions, value:int) -> None:
        state = self.loc_to_state(loc, self.n)
        q_table[state-1, Actions.get_index(action)] = value

    def get_optimal_policy(self) :
        return [Actions.index_to_action(index) for index in np.nanargmax(self.q_table, axis=1)] # ! again in this part too, use `nanargmax`
    
    def show_path(self) :
        current_state = 1
        optimal_policy = self.get_optimal_policy()
        while current_state != (self.n**2) :
            loc = self.state_to_loc(current_state, self.n)
            action = optimal_policy[current_state-1]
            print(current_state, action)

            current_state = self.loc_to_state(self.move(loc, action), self.n)


    def _write_logs(self, loc, best_action, action, new_value, epsilon) :
            # print(loc)
            # print(best_action, action)
            # print(new_value)
            # print(self.q_table[self.loc_to_state(loc, self.n)-1, :], end="\n\n")

            # Logging  
            self.logger.info("Location: %s", loc)
            if self.epsilon_greedy : 
                self.logger.info("epsilon-0.7: %s", epsilon-self.epsilon_discount_factor)
            self.logger.info("Best action: %s\nAction taken: %s", best_action, action) 
            self.logger.info(new_value)
            self.logger.info("%s\n", self.q_table[self.loc_to_state(loc, self.n)-1, :].tolist())
            
        
    def run(self, n_iter=50) :
        start_loc = (0, 0)
        loc = start_loc
        if self.epsilon_greedy :
            epsilon_values = QLearning.minmax_scale(np.arange(n_iter))[::-1]

        for iteration_num in range(n_iter) :
            if self.epsilon_greedy :
                epsilon = epsilon_values[iteration_num]
            else :
                epsilon = 0
            actions = self.get_actions(loc)

            best_action = self.get_best_move(loc, actions)

            # with p=0.7 the best action will happen and with p=0.3 the reverse of it
            action = self.perturb_action(loc, best_action, actions, epsilon)


            new_loc = self.move(loc, action)
            if new_loc == -1 : # means we have reached the terminal
                new_value = self.bellman(loc, new_loc, action)
                self.update_value(self.q_table, loc, action, new_value)
                new_loc = (0, 0)
                self.complete_cycle += 1
                
            else :
                new_value = self.bellman(loc, new_loc, action) # new value for the current q-value
                self.update_value(self.q_table, loc, action, new_value) # FIXME it doesn't update , check the log of this block output

            if self.log :
                self._write_logs(loc, best_action, action, new_value, epsilon)
            loc = new_loc


    def __run_simulation(self, n_iter=50) :
        '''yields loc at each iteration.
        may not be updated as the main run()
        '''
        start_loc = (0, 0)
        loc = start_loc
        if self.epsilon_greedy :
            epsilon_values = QLearning.minmax_scale(np.arange(n_iter))[::-1]

        for iteration_num in range(n_iter) :
            if self.epsilon_greedy :
                epsilon = epsilon_values[iteration_num]
            else :
                epsilon = 0
            actions = self.get_actions(loc)

            best_action = self.get_best_move(loc, actions)

            # with p=0.7 the best action will happen and with p=0.3 the reverse of it
            action = self.perturb_action(loc, best_action, actions, epsilon)


            new_loc = self.move(loc, action)
            if new_loc == -1 : # means we have reached the terminal
                new_value = self.bellman(loc, new_loc, action)
                self.update_value(self.q_table, loc, action, new_value)
                new_loc = (0, 0)
                self.complete_cycle += 1
                
            else :
                new_value = self.bellman(loc, new_loc, action) # new value for the current q-value
                self.update_value(self.q_table, loc, action, new_value) # FIXME it doesn't update , check the log of this block output

            if self.log :
                self._write_logs(loc, best_action, action, new_value, epsilon)
            loc = new_loc
            yield loc
