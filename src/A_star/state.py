class State:
    def __init__(self, state_number, h=0, g=0, parent=None) -> None:
        self.state_number = state_number
        self.h = h
        self.g = g
        self.parent = parent

    def get_f(self) :
        return (self.h + self.g)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{self.state_number}"

    def __eq__(self, other) -> bool:
        return self.get_f() == other.get_f()

    def __ne__(self, other) -> bool:
        return self.get_f() != other.get_f()
    
    def __lt__(self, other) -> bool:
        return self.get_f() < other.get_f()
    
    def __gt__(self, other) -> bool:
        return self.get_f() > other.get_f()
