from typing import List, Tuple
import numpy as np
import logging

from src.RL.q_learning.action import Actions
from src.RL.q_learning import QLearning

class Approximate_Qlearning() :
    def __init__(
            self, n, snakes:List[Tuple[int, int]], ladders:List[Tuple[int, int]],
            alpha=0.3, p=0.7, lambda_=1, log=True
        ) :
        self.n = n
        self.snakes = snakes
        self.ladders = ladders
        self.snakes_loc = []
        self.ladders_loc = []
        self.alpha = alpha
        self.p = p
        self.lambda_ = lambda_
        self.features = [self.f1, self.f2, self.f3]
        self.weights = np.random.randn(3) # .reshape(1, 2) # ! if you add more features later, you should change it too. (i know, BAD CODE :)
        self.log = log

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

        # configure logging
        if log :  # FIXME ! the address of the log file is incorrect
            logging.basicConfig(filename='../../logs/approximate_q_learning_log.txt', level=logging.DEBUG, format='%(message)s')
            self.logger = logging.getLogger()

    def inbound(self, loc, x=0, y=0) :
        new_loc = (loc[0]+y, loc[1]+x)
        if (0 <= new_loc[0] < self.n) and (0 <= new_loc[1] < self.n) :
            return True
        return False
    
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

    def get_actions(self, loc) :
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
        
    def get_best_move(self, state) -> List[Tuple[Actions, int]]:
        '''returns:
            - best action 
            - predicted value of that action
        '''
        actions = self.get_actions(self.state_to_loc(state, self.n))
        action_value = {}
        for action in actions :
            value = self.predict(state, action)
            action_value[action] = value

        best_action = sorted(action_value, key=lambda x: action_value[x])[-1]
        return best_action, action_value[best_action]

    def predict(self, state, action) :
        # features_prediction = np.vectorize(lambda f: f(state, action))(features) 
        features_prediction = 0
        for index, f in enumerate(self.features) :
            w = self.weights[index]
            # print(w)
            features_prediction += w*f(state, action)
        
        return features_prediction

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
    
    def f1(self, state, action:Actions) : 
        '''distance from the final state'''
        loc = self.state_to_loc(state, self.n)
        new_loc = self.move(loc, action)
        new_state = self.loc_to_state(new_loc, self.n)
        return -1*(new_state / self.n**2)
    
        # loc = self.state_to_loc(state, self.n)
        # new_loc = self.move(loc, action)
        # distance = ((new_loc[0] - (self.n-1))**2 + (new_loc[1] - (self.n-1))**2) ** 0.5 # distance from the final state
        # return (1- distance)

        # loc = self.state_to_loc(state, self.n)
        # new_loc = self.move(loc, action)
        # new_state = self.loc_to_state(new_loc, self.n)
        # return (self.n**2) - new_state


    def f2(self, state, action:Actions) :
        '''distance from a ladder'''
        # loc = self.state_to_loc(state, self.n)
        # new_loc = self.move(loc, action)
        # state = self.loc_to_state(new_loc, self.n)

        # min_distance = self.n**2
        # for ladder in self.ladders :
        #     ladder_bottom = ladder[0]
        #     if (diff:= (ladder_bottom - state)) < min_distance and (diff > 0):
        #         min_distance = ladder_bottom - state
        
        # return 1 - (min_distance / self.n**2 )

        loc = self.state_to_loc(state, self.n)
        new_loc = self.move(loc, action)
        state = self.loc_to_state(new_loc, self.n)
        for x, y in self.ladders_loc[:, 0:2] :
            if (x==new_loc[0]) and (y==new_loc[1]) :
                return 2
        return 0

    def f3(self, state, action:Actions) :
        '''distance from a snake'''
        # loc = self.state_to_loc(state, self.n)
        # new_loc = self.move(loc, action)
        # state = self.loc_to_state(new_loc, self.n)

        # min_distance = self.n**2
        # for snake in self.snakes :
        #     snake_head = snake[1]
        #     if (diff:= (snake_head - state)) < min_distance and (diff > 0):
        #         min_distance = snake_head - state
        
        # return (min_distance / self.n**2 )

        loc = self.state_to_loc(state, self.n)
        new_loc = self.move(loc, action)
        state = self.loc_to_state(new_loc, self.n)
        for x, y in self.snakes_loc[:, 0:2]  :
            if (x==new_loc[0]) and (y==new_loc[1]) :
                return -2
        return 0
            
    def update_weights(self, difference, state, action) :
        # features_prediction = np.vectorize(lambda f: f(state, action))(features) 
        for index, f in enumerate(self.features) :
            w = self.weights[index]
            prediction = w*f(state, action)
            self.weights[index] = self.weights[index] + self.alpha*difference*prediction

    def show_path(self) :
        state = 1
        loc = (0, 0)
        while state != (self.n**2) :
            action, _ = self.get_best_move(state)
            print(state, action)
            loc = self.move(loc, action)
            state = self.loc_to_state(loc, self.n)

    def train(self, n_iter=50) :
        start_loc = (0, 0)
        loc = start_loc

        for iteration_num in range(n_iter) :
            current_state = self.loc_to_state(loc, self.n)
            action, predicted_value = self.get_best_move(current_state)
            # action = self.perturb_action(loc, best_action, actions, epsilon)

            # now perform the action
            new_loc = self.move(loc, action)
            new_state = self.loc_to_state(new_loc, self.n)
            _, next_state_value = self.get_best_move(new_state)
            reward = Actions.get_reward(action)
            difference = (reward + self.lambda_*next_state_value) - predicted_value

            self.update_weights(difference, current_state, action)


    