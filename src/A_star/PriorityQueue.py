class PriorityQueue:
    def __init__(self):
        self.queue = []

    def put(self, state_object):
        self.queue.append(state_object)
        self.queue.sort(reverse=True, key=lambda s: s.get_f())  # Sort in descending order based on priority

    def get_highest_priority(self):
        if self.queue:
            return self.queue[-1]  # Return the item with the highest priority
        raise IndexError("Priority queue is empty")

    def pop_highest_priority(self):
        if self.queue:
            return self.queue.pop()  # Remove and return the item with the highest priority
        raise IndexError("Priority queue is empty")

    def is_empty(self):
        return len(self.queue) == 0
    
    # def __repr__(self) -> str:
    #     f"{self.queue}"