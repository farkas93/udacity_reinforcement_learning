import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = 0.005
        self.gamma = 1.0
        self.alpha = .05

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        
        if np.random.random() > self.epsilon:
            return np.argmax(self.Q[state])
        else:
            return np.random.choice(np.arange(self.nA))
    
    def get_probs(self, state):
        """ Given the state, return the probabilities for chosing any of the possible actions.

        Params
        ======
        - state: the current state of the environment.

        Returns
        =======
        - probs: list of float type probabilities for chosing each of the possible actions.
        """
        probs = np.ones(self.nA) * self.epsilon / self.nA
        max_ind = np.argmax(self.Q[state])
        probs[max_ind] += (1.0 - self.epsilon)
        return probs
    
    def q_learn(self, next_state):
        next_action = np.argmax(self.Q[next_state])
        return self.Q[next_state][next_action]
        
    def expected_sarsa(self, next_state):
        return np.dot(self.get_probs(next_state),self.Q[next_state])
        
    
    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        q_cur = self.Q[state][action]
        q_update = self.q_learn(next_state)
        self.Q[state][action] = q_cur + self.alpha * (reward + self.gamma*q_update - q_cur)
        