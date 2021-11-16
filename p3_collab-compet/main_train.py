from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt
from agents.ddpg_agent import MADDPGCollective
import csv
import sys
import pandas as pd

import torch

TARGET_SCORE = 0.5
USE_COLLECTIVE_TRAINING = True

def plot_scores(scores, episode_solved, rolling_window=100):
    """Plot scores and optional rolling mean using specified window."""
    sums  = []
    for s in scores:
        sums.append(np.sum(s))
    rolling_mean = pd.Series(sums).rolling(rolling_window).mean()       
    # plot the scores
    fig = plt.figure()
    plt.title("Scores")
    plt.plot(np.arange(len(sums)), sums, label="Score of both agents / episode")
    plt.plot(rolling_mean, label="AVG-Score over previous 100 episodes")
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.axhline(y = rolling_mean[len(rolling_mean)-1], color = 'r', linestyle = '-', label="AVG-Score last Epoch")
    plt.axhline(y = TARGET_SCORE, color = 'g', linestyle = 'dashed', label="Target Score")   
    if episode_solved > -1:
        plt.axvline(x = episode_solved, color = 'b', linestyle = '-', label="Episode Environment Solved")
    plt.legend(bbox_to_anchor = (0.12, 1.01), loc = 'upper center')
    plt.show()
    pass


def write_to_csv(scores, episode_solved, model_name):

    with open("log_"+model_name+".csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow("Episode Solving the Problem;{}".format(episode_solved))
        writer.writerow("Scores")
        writer.writerows(scores)
    pass

def main():    
    args = sys.argv[1:]
    if len(args) > 0:
        model_name = args[0]
    else:
        model_name = "unnamed_model"
    print("Start training the model named: {}".format(model_name))

    env = UnityEnvironment(file_name='./Tennis/Tennis.exe')

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
    agent_collective = MADDPGCollective(device, num_agents, state_size=state_size, action_size=action_size, random_seed=47)

    # Train the DQN agent
    scores, episode_solved = agent_collective.train(env, brain_name, TARGET_SCORE)

    #Save the trained model
    for agent_ind in range(num_agents):
        agent_collective.save_current_model('./saved_models/', model_name, agent_ind)
    
    write_to_csv(scores, episode_solved, model_name)
    plot_scores(scores, episode_solved)


if __name__ == "__main__":
    main()