from unityagents import UnityEnvironment
import numpy as np
import torch
import matplotlib.pyplot as plt
from agents.dqn_agent import DQNAgent
import sys
import pandas as pd


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
    plt.show()
    pass

def main():    
    args = sys.argv[1:]
    if len(args) > 0:
        model_name = args[0]
    else:
        model_name = "unnamed_model"
    print("Start training the model named: {}".format(model_name))

    env = UnityEnvironment(file_name="./Banana/Banana.exe")
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space 
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)


    agent = DQNAgent(state_size=37, action_size=4, seed=0)

    # Train the DQN agent
    scores = agent.train(env, brain_name, n_episodes=1800, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995)

    #Save the trained model
    torch.save(agent.qnetwork_local.state_dict(), './saved_models/'+ model_name +'.pth')


if __name__ == "__main__":
    main()