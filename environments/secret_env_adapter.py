import numpy as np


class SecretEnvAdapter:
   
    def __init__(self, secret_env):
        self.env = secret_env
        self.states = list(range(secret_env.num_states()))
        self.actions = list(range(secret_env.num_actions()))
        self._last_score = 0.0
        self._model_cache = {}        

    def reset(self):
        self.env.reset()
        self._last_score = self.env.score()
        return self.env.state_id()

    def step(self, action):
        self.env.step(action)
        new_score = self.env.score()
        reward = new_score - self._last_score
        self._last_score = new_score
        return self.env.state_id(), reward, self.env.is_game_over()

    def available_actions(self):
        return [int(a) for a in self.env.available_actions()]

    def transition_prob(self, s, a):
       
        if (s, a) in self._model_cache:          # déjà calculé ? on relit
            return self._model_cache[(s, a)]

        transitions = []                          # sinon on calcule
        for s_p in self.states:
            for r_index in range(self.env.num_rewards()):
                prob = self.env.p(s, a, s_p, r_index)
                if prob > 0.0:
                    transitions.append((prob, s_p, self.env.reward(r_index)))

        self._model_cache[(s, a)] = transitions   # et on mémorise
        return transitions

    def is_terminal(self, state):
        for a in self.actions:
            if self.transition_prob(state, a):   
                return False
        return True

    def render(self):
        self.env.display()