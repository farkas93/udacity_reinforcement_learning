import numpy as np
import random
from collections import deque
import time

from models.linear_model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

from agents.utility import OUNoise, ReplayBuffer

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
LEARN_AFTER_TS = 5     # learn after a given amount of time steps
NR_UPDATES = 10         # sample experience and learn from it this many times

EPOCHS = 20
EPISODES_PER_EPOCH = 100
TIMESTEPS_PER_EPISODE = 500
TOTAL_EPISODES = EPOCHS * EPISODES_PER_EPOCH

GRAD_CLIP_THRESHOLD = 1.0

FLOAT_PRINT_FORMAT = "{:.4f}"

class DDPGAgent():

    def __init__(self, state_size, action_size, random_seed, device):
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        pass


class MADDPGCollective():
    """Interacts with and learns from the environment."""
    
    def __init__(self, device, num_agents, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.epsilon_noise = 1.0
        self.device = device
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        
        self.agents = []
        for i in range(self.num_agents):
            self.agents.append(DDPGAgent(state_size, action_size, random_seed, device))
            self.hard_update(self.agents[i].actor_local, self.agents[i].actor_target)
        

        
        # Replay memory
        self.memory = ReplayBuffer(device, action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)
    
    def step(self, states, actions, rewards, next_states, dones, timestep):
        """Save experience in replay memory, and use random sample from buffer to learn."""

        # Save experience / reward
        for agent_ind in range(self.num_agents):    
            self.memory.add(states[agent_ind], actions[agent_ind], rewards[agent_ind], next_states[agent_ind], dones[agent_ind])

            # Learn, if enough samples are available in memory
            if len(self.memory) > BATCH_SIZE and timestep % LEARN_AFTER_TS == 0:            
                for _ in range(NR_UPDATES):               
                    experiences = self.memory.sample() 
                    self.learn(experiences, agent_ind, GAMMA)

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(states).float().to(self.device)
        actions = np.zeros((self.num_agents, self.action_size))
        for agent_ind in range(self.num_agents):
            self.agents[agent_ind].actor_local.eval()
            with torch.no_grad():
                actions[agent_ind, :] = self.agents[agent_ind].actor_local(state).cpu().data.numpy()[agent_ind, :]
            self.agents[agent_ind].actor_local.train()
            if add_noise:
                actions[agent_ind, :] += self.epsilon_noise * self.noise.sample()
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, agent_ind, gamma):
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
        actions_next = self.agents[agent_ind].actor_target(next_states)
        Q_targets_next = self.agents[agent_ind].critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.agents[agent_ind].critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.agents[agent_ind].critic_optimizer.zero_grad()
        critic_loss.backward()
        # gradient clipping for critic
        if GRAD_CLIP_THRESHOLD > 0:
            torch.nn.utils.clip_grad_norm_(self.agents[agent_ind].critic_local.parameters(), GRAD_CLIP_THRESHOLD)
        self.agents[agent_ind].critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.agents[agent_ind].actor_local(states)
        actor_loss = -self.agents[agent_ind].critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.agents[agent_ind].actor_optimizer.zero_grad()
        actor_loss.backward()
        self.agents[agent_ind].actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.agents[agent_ind].critic_local, self.agents[agent_ind].critic_target, TAU)
        self.soft_update(self.agents[agent_ind].actor_local, self.agents[agent_ind].actor_target, TAU)

        self.epsilon_noise *= 0.9
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

    def hard_update(self, local_model, target_model):
        """Copy network parameters from local_model to target_model
            Params
            ======
                local_model: PyTorch model (weights will be copied from)
                target_model: PyTorch model (weights will be copied to)
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

    def save_current_model(self, path, model_name, agent_ind):
        """Save the models of the actor and the critic network which belong to the
           training model called model_name to the path.
        
        Params
        ======
            model_name: name of the current model which is being trained
            path: path where we want to save the 
        
        """
        torch.save(self.agents[agent_ind].actor_local.state_dict(), path + model_name + '_actor_agent{}.pth'.format(agent_ind))
        torch.save(self.agents[agent_ind].critic_local.state_dict(), path + model_name + '_critic_agent{}.pth'.format(agent_ind))
        torch.save(self.agents[agent_ind].actor_target.state_dict(), path + model_name + '_targetactor_agent{}.pth'.format(agent_ind))
        torch.save(self.agents[agent_ind].critic_target.state_dict(), path + model_name + '_targetcritic_agent{}.pth'.format(agent_ind))
        pass

    
    def load_model(self, actor, critic, target_actor, target_critic):
        """Save the models of the actor and the critic network which belong to the
           training model called model_name to the path.
        
        Params
        ======
            model_name: name of the current model which is being trained
            path: path where we want to save the 
        
        """
        for agent_ind in range(self.num_agents):
            suffix = '_agent{}.pth'.format(agent_ind)
            self.agents[agent_ind].actor_local.load_state_dict(torch.load(actor+suffix)) 
            self.agents[agent_ind].critic_local.load_state_dict(torch.load(critic+suffix))
            self.agents[agent_ind].actor_target.load_state_dict(torch.load(target_actor+suffix))
            self.agents[agent_ind].critic_target.load_state_dict(torch.load(target_critic+suffix))
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
        solved = False
        episode_env_solved = -1
        for i_episode in range(1, TOTAL_EPISODES+1):
            env_info = env.reset(train_mode=True)[brain_name]     # reset the environment 
            states = env_info.vector_observations                 # get the current state (for each agent)
            self.reset()        
            scores = np.zeros(self.num_agents)                    # initialize the score (for each agent)
            dones = np.zeros(self.num_agents) 
            start_time = time.time()
            t = 0
            for t in range(TIMESTEPS_PER_EPISODE):
                actions = self.act(states)
                
                env_info = env.step(actions)[brain_name]           # send all actions to tne environment
                next_states = env_info.vector_observations         # get next state (for each agent)
                rewards = env_info.rewards                         # get reward (for each agent)
                dones = env_info.local_done                        # see if episode finished
                self.step(states, actions, rewards, next_states, dones, t)

                states = next_states
                scores += rewards

                if np.any(dones):
                    break 
            
            duration_episode = time.time() - start_time
            #Store the scores
            total_score = np.sum(scores)
            scores_deque.append(total_score)
            all_scores.append(scores)

            #Print for overview and check if environment solved
            score_over_last_epoch = np.mean(scores_deque)
            
            if score_over_last_epoch >= target_score and not solved:
                episode_env_solved = i_episode
                solved = True
            print(('\rEpisode {}\tAverage Score: '+FLOAT_PRINT_FORMAT+' \tEpisode duration {}s ({} timesteps) \tEstimated time left: {}s, scores of the agents: {}').format(i_episode, score_over_last_epoch, round(duration_episode), t, round((TOTAL_EPISODES-i_episode)*duration_episode), scores), end="")
            if i_episode % EPISODES_PER_EPOCH == 0:
                for agent_ind in range(self.num_agents):
                    self.save_current_model('','checkpoint', agent_ind)
                print(('\rEpisode {}\tAverage Score: '+FLOAT_PRINT_FORMAT).format(i_episode, score_over_last_epoch))
            if solved and total_score > 1.5:
                print("Environment Solved after {} Epochs and {} Episodes! Epsilon noise was: {}".format(i_episode/EPISODES_PER_EPOCH, episode_env_solved, self.epsilon_noise))
                break
                
        return all_scores, episode_env_solved

