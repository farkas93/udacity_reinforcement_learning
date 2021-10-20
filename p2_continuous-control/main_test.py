from unityagents import UnityEnvironment
import numpy as np
import torch
import matplotlib.pyplot as plt
from agents.ddpg_agent import DDPGAgentCollective
import sys


TARGET_SCORE = 30
USE_COLLECTIVE_TRAINING = True

def main():  
    
    args = sys.argv[1:]
    if len(args) > 0:
        model_name = args[0]
        actor_to_test = "./saved_models/"+model_name+"_actor.pth"
        critic_to_test = "./saved_models/"+model_name+"_critic.pth"
    else:
        model_name = "checkpoint"
        actor_to_test = "./checkpoint_actor.pth"
        critic_to_test = "./checkpoint_critic.pth"
    
    print("Testing the model named: {}".format(model_name))

    if USE_COLLECTIVE_TRAINING:
        env = UnityEnvironment(file_name='./Reacher/Reacher.exe')
    else:
        env = UnityEnvironment(file_name='./Reacher_single/Reacher.exe')

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    env_info = env.reset(train_mode=False)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents: {}'.format(num_agents))

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action: {}'.format(action_size))

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('State(s) look like: {}'.format(states.shape))
    print('One state has the length: {}'.format(state_size))


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent_collective = DDPGAgentCollective(device, num_agents, state_size=state_size, action_size=action_size, random_seed=47)
    agent_collective.actor_local.load_state_dict(torch.load(actor_to_test))
    agent_collective.critic_local.load_state_dict(torch.load(critic_to_test))

    

    # load the weights from file and test the agent
    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    while True:
        actions = agent_collective.act(states, add_noise=False)
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

if __name__ == "__main__":
    main()