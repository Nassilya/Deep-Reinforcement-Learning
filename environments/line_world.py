import numpy as np

class LineWorld:
    """
    Line World : une ligne de cases, cases extrêmes = terminales
    Case de droite -> reward +1, case de gauche -> reward -1
    """

    def __init__(self, size=5):
        self.size = size                          
        self.states = list(range(size))           
        self.actions = [0, 1]                      
        self.terminal_states = [0, size - 1]       
        self.agent_pos = None                      

    def reset(self):
        self.agent_pos = self.size // 2            
        return self.agent_pos

    def is_terminal(self, state):
        return state in self.terminal_states

    def step(self, action):
        if action == 0:                            
            self.agent_pos -= 1
        else:                                      
            self.agent_pos += 1

        reward = 0.0
        done = self.is_terminal(self.agent_pos)
        if self.agent_pos == self.size - 1:        
            reward = 1.0
        elif self.agent_pos == 0:                  
            reward = -1.0

        return self.agent_pos, reward, done

    def transition_prob(self, s, a):  
        if self.is_terminal(s):                    
            return [(1.0, s, 0.0)]

        s_next = s - 1 if a == 0 else s + 1        
        reward = 0.0
        if s_next == self.size - 1:
            reward = 1.0
        elif s_next == 0:
            reward = -1.0
        return [(1.0, s_next, reward)]

    def render(self):
        line = ""
        for i in range(self.size):
            if i == self.agent_pos:
                line += "[A]"                      
            elif i in self.terminal_states:
                line += "[T]"                      
            else:
                line += "[ ]"
        print(line)