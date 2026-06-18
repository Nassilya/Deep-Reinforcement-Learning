import numpy as np


class GridWorld:
    """
    Grille 2D 5x5 
    Agent : départ en haut à gauche (case 0)
    Piège : coin haut-droit (case 4), récompense -3
    But   : coin bas-droit (case 24), récompense +1
    Actions : 0=haut, 1=bas, 2=gauche, 3=droite
    """

    def __init__(self, rows=5, cols=5):
        self.rows = rows
        self.cols = cols
        self.states = list(range(rows * cols))     
        self.actions = [0, 1, 2, 3]                 
        self.trap = cols - 1                        
        self.goal = rows * cols - 1                 
        self.terminal_states = [self.trap, self.goal]
        self.agent_pos = None

    def reset(self):
        self.agent_pos = 0
        return self.agent_pos

    def is_terminal(self, state):
        return state in self.terminal_states

    def _state_to_rc(self, state):
        return state // self.cols, state % self.cols

    def _rc_to_state(self, row, col):
        return row * self.cols + col

    def _next_position(self, state, action):
        row, col = self._state_to_rc(state)
        if action == 0:                              
            row = max(row - 1, 0)
        elif action == 1:                            
            row = min(row + 1, self.rows - 1)
        elif action == 2:                            
            col = max(col - 1, 0)
        elif action == 3:                            
            col = min(col + 1, self.cols - 1)
        return self._rc_to_state(row, col)

    def step(self, action):
        self.agent_pos = self._next_position(self.agent_pos, action)
        reward = 0.0
        done = self.is_terminal(self.agent_pos)
        if self.agent_pos == self.goal:
            reward = 1.0
        elif self.agent_pos == self.trap:
            reward = -3.0
        return self.agent_pos, reward, done

    def transition_prob(self, s, a):
        if self.is_terminal(s):
            return [(1.0, s, 0.0)]
        s_next = self._next_position(s, a)
        reward = 0.0
        if s_next == self.goal:
            reward = 1.0
        elif s_next == self.trap:
            reward = -3.0
        return [(1.0, s_next, reward)]

    def render(self):
        for row in range(self.rows):
            line = ""
            for col in range(self.cols):
                s = self._rc_to_state(row, col)
                if s == self.agent_pos:
                    line += "[A]"
                elif s == self.goal:
                    line += "[G]"
                elif s == self.trap:
                    line += "[X]"
                else:
                    line += "[ ]"
            print(line)
        print()