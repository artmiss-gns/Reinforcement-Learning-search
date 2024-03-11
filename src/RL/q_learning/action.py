class Actions:
    _actions = ["up", "right_1", "right_2", "left_1", "left_2", "down", "ladder_up", "snake_down", "terminate"]
    up, right_1, right_2, left_1, left_2, down, ladder_up, snake_down, terminate = _actions

    @classmethod
    def get_index(cls, action):
        return cls._actions.index(action)

    @classmethod
    def index_to_action(cls, index:int) :
        return cls._actions[index]
    
    @classmethod
    def get_reward(cls, action) :
        if action in ["up", "right_1", "left_1", "down"] :
            return -1
        elif action in ["left_2", "right_2"]:
            return -1.5 # value -2 for this doesn't make sense at all!
        elif action in ["terminate"] :
            return 100
        elif action in ["ladder_up"] : 
            return 2
        elif action in ["snake_down"] :
            return -2
            
    @classmethod    
    def get_num_actions(cls):
        return len(cls._actions)