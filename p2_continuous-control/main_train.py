from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt
from agents.ddpg_agent import DDPGAgentCollective
import sys
import pandas as pd

import torch

TARGET_SCORE = 30
USE_COLLECTIVE_TRAINING = True

def plot_scores(scores, rolling_window=100):
    """Plot scores and optional rolling mean using specified window."""
    plt.title("Scores")
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()    
    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.plot(rolling_mean)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.axhline(y = rolling_mean, color = 'r', linestyle = '-', label="my avg score")
    plt.axhline(y = TARGET_SCORE, color = 'g', linestyle = 'dashed', label="target score")    
    plt.legend(bbox_to_anchor = (1.0, 1), loc = 'upper center')
    plt.show()
    pass

def main():    
    args = sys.argv[1:]
    if len(args) > 0:
        model_name = args[0]
    else:
        model_name = "unnamed_model"
    print("Start training the model named: {}".format(model_name))

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
    print('Training on Device: {}'.format(device))
    agent_collective = DDPGAgentCollective(device, num_agents, state_size=state_size, action_size=action_size, random_seed=47)

    # Train the DQN agent
    scores = agent_collective.train(env, brain_name, TARGET_SCORE)

    #Save the trained model
    agent_collective.save_current_model('./saved_models/', model_name)

    plot_scores(scores)


if __name__ == "__main__":
    main()