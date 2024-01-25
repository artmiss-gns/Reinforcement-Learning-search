from src.A_star.PriorityQueue import PriorityQueue
from src.A_star.state import State
from src.RL.action import Actions

from typing import Tuple, List
import numpy as np

class A_star :
    def __init__(self, start_state, n, snakes, ladders) -> None:
        self.start_state = start_state
        self.visited = [] # contains the `state_number` of each state that we visit
        self.check_queue = PriorityQueue()
        self.current_state = start_state
        self.n = n
        self.snakes = snakes
        self.snakes_loc = []
        self.ladders= ladders
        self.ladders_loc = []

        # self.preprocess()
        
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
        
        self.snakes_loc = np.array(self.snakes_loc)
        self.ladders_loc = np.array(self.ladders_loc)
        
    def get_heuristic(self, state: State) :
        loc = self.state_to_loc(state.state_number, self.n)

        if self.snakes and list(loc) in self.snakes_loc[:, -2:].tolist(): # if we are at the head of a snake
            for bottom, top  in self.snakes :
                if top == state.state_number :
                    loc = self.state_to_loc(bottom, self.n)
                    break
        elif self.ladders and list(loc) in self.ladders_loc[:, 0:2].tolist(): # if we are at the bottom of a ladder
            # we are currently at the bottom of the ladder, so we get it's top :
            for bottom, top  in self.ladders :
                if bottom == state.state_number :
                    loc = self.state_to_loc(top, self.n)
                    break
        
        state_number = self.loc_to_state(loc, self.n)
        return ((self.n**2)-1) - state_number
        # return  ((self.n-1) - loc[0]) + loc[1] 
    
    def preprocess(self) :
        # assigning values to States
        n = self.n
        parent = None
        for state_number in range(1, (n**2)+1) :
            state = State(state_number=state_number, parent=parent, g=state_number-1, h=self.heuristic(state))
            self.states[state.state_number]: state
            parent = state

    def get_actions(self, state: State) -> List[Actions]:
        n = self.n
        loc = self.state_to_loc(state.state_number, n)
        if self.loc_to_state(loc, n) == n**2 : # final state
            return [Actions.terminate]
        elif self.snakes and list(loc) in self.snakes_loc[:, -2:].tolist(): # if we are at the head of a snake
            return [Actions.snake_down]
        elif self.ladders and list(loc) in self.ladders_loc[:, 0:2].tolist(): # if we are at the bottom of a ladder
            return [Actions.ladder_up]
        else :
            # ! if we could NOT move up in all the side states, uncomment this part :
            actions = []
            if loc[1] + 1 < n:
                actions.append(Actions.right_1)
            if loc[1] + 2 < n :
                actions.append(Actions.right_2)
            if loc[1] - 1 >= 0 :
                actions.append(Actions.left_1)
            if loc[1] - 2 >= 0 :
                actions.append(Actions.left_2)

            if (loc[0]%2 == 0) and (loc[1] == (self.n-1)) : # even rows
                actions.extend([Actions.up]) #  [..,Actions.down] if we could come down from sides
            elif (loc[0]%2 != 0) and (loc[1] == (0)) : # odd rows
                actions.extend([Actions.up]) #  [..,Actions.down] if we could come down from sides
                
            return actions
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

    def inbound(self, loc, x=0, y=0) :
        new_loc = (loc[0]+y, loc[1]+x)
        if (0 <= new_loc[0] < self.n) and (0 <= new_loc[1] < self.n) :
            return True
        return False
    
    def move(self, loc, action) -> List[ Tuple[int, int]]:
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
    
    def get_neighbors(self, parent:State) -> List[State]:
        '''returns state number'''
        actions = self.get_actions(parent)
        neighbors = []
        for action in actions :
            neighbor_loc = self.move(self.state_to_loc(parent.state_number, self.n), action)
            neighbor_state_number = self.loc_to_state(neighbor_loc, self.n)
            neighbor = State(
                state_number=neighbor_state_number,
                g=parent.g+1,
                parent=parent
            )
            neighbor.h = self.get_heuristic(neighbor)

            neighbors.append(neighbor)
        
        return neighbors
    
    def get_best_state(self) -> State:
        return self.check_queue.pop_highest_priority()

    def run(self) :
        current_state = State(self.start_state, g=0, parent=None)
        current_state.h = self.get_heuristic(current_state)
        while current_state.state_number != (self.n**2) :
            print(current_state.state_number)
            neighbors = self.get_neighbors(current_state)
            for n in neighbors :
                if n.state_number not in self.visited :                    
                    self.check_queue.put(n)
                    
            self.visited.append(current_state.state_number)
            next_state = self.get_best_state()
            next_state.parent = current_state
            current_state = next_state # update the current state
        print(current_state.state_number)