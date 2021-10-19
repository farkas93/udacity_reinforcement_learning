import numpy as np
import random
from collections import deque

from models.linear_model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

from agents.utility import OUNoise, ReplayBuffer

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3        # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
LEARN_AFTER_TS = 20 #lear after a given amount of time steps

EPOCHS = 1
EPISODES_PER_EPOCH = 100
TIMESTEPS_PER_EPISODE = 1000
TOTAL_EPISODES = EPOCHS * EPISODES_PER_EPOCH


GRAD_CLIP_THRESHOLD = 1.0 #TODO: why gradient clipping on value 1?


class DDPGAgentCollective():
    """Interacts with and learns from the environment."""
    
    def __init__(self, device, num_agents, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.epsilon = 1.0
        self.device = device
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(device, action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def step(self, states, actions, rewards, next_states, dones, timestep):
        """Save experience in replay memory, and use random sample from buffer to learn."""

        #for i in range(len(states)):
        # Save experience / reward
        self.memory.add(states, actions, rewards, next_states, dones)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and timestep % LEARN_AFTER_TS == 0:            
            for _ in range(LEARN_AFTER_TS):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(states).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.epsilon * self.noise.sample() #TODO: why epsilon decay on the noise?
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()        
        # gradient clipping for critic
        if GRAD_CLIP_THRESHOLD > 0:
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), GRAD_CLIP_THRESHOLD)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

        self.epsilon *= 0.99
        self.noise.reset()                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save_current_model(self, model_name, path):
        """Save the models of the actor and the critic network which belong to the
           training model called model_name to the path.
        
        Params
        ======
            model_name: name of the current model which is being trained
            path: path where we want to save the 
        
        """
        torch.save(self.actor_local.state_dict(), path + model_name + '_actor.pth')
        torch.save(self.critic_local.state_dict(), path + model_name + '_critic.pth')
        pass

    def train(self, env, brain_name, target_score):
        """Save the models of the actor and the critic network which belong to the
           training model called model_name to the path.
        
        Params
        ======
            env: game environment
            brain_name: name of the unity brain
            target_score: average score we require the agent to get so the environment is treated as solved 
        """
        scores_deque = deque(maxlen=EPISODES_PER_EPOCH)
        all_scores = []

        for i_episode in range(1, TOTAL_EPISODES+1):
            env_info = env.reset(train_mode=True)[brain_name]     # reset the environment    
            states = env_info.vector_observations                  # get the current state (for each agent)
            self.reset()        
            scores = np.zeros(self.num_agents)                          # initialize the score (for each agent)
            for t in range(TIMESTEPS_PER_EPISODE):
                actions = self.act(states)
                env_info = env.step(actions)[brain_name]           # send all actions to tne environment
                next_states = env_info.vector_observations         # get next state (for each agent)
                rewards = env_info.rewards                         # get reward (for each agent)
                dones = env_info.local_done                        # see if episode finished

                for i in range(len(states)):
                    self.step(states[i], actions[i], rewards[i], next_states[i], dones[i], t)

                states = next_states
                scores += rewards
                if np.any(dones):
                    break 

            #Store the scores
            scores_deque.append(scores)
            all_scores.append(scores)

            #Print for overview and check if environment solved
            score_over_last_epoch = np.mean(scores_deque)
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, score_over_last_epoch), end="")
            if i_episode % EPISODES_PER_EPOCH == 0:
                torch.save(self.actor_local.state_dict(), 'checkpoint_actor.pth')
                torch.save(self.critic_local.state_dict(), 'checkpoint_critic.pth')
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, score_over_last_epoch))
                if score_over_last_epoch >= target_score:
                    print("Environment Solved after {} Episodes!".format(i_episode))
                    break
                
        return all_scores

