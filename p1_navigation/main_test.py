from unityagents import UnityEnvironment
import numpy as np
import torch
import matplotlib.pyplot as plt
from agents.dqn_agent import DQNAgent
import sys


def main():  
    
    args = sys.argv[1:]
    if len(args) > 0:
        model_name = args[0]
        model_to_test = "./saved_models/"+model_name+".pth"
    else:
        model_name = "checkpoint"
        model_to_test = "./checkpoint.pth"
    print("Start training the model named: {}".format(model_name))

    env = UnityEnvironment(file_name="./Banana/Banana.exe")
    # get the default brain
    brain_name = env.brain_names[0]

    agent = DQNAgent(state_size=37, action_size=4, seed=0)

    

    # load the weights from file and test the agent
    agent.qnetwork_local.load_state_dict(torch.load(model_to_test))
    scores = []
    episode_cnt = 0
    while episode_cnt < 100:
        env_info = env.reset(train_mode=False)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        score = 0                                          # initialize the score
        while True:
            action = int(agent.act(state))        # select an action
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            score += reward                                # update the score
            state = next_state                             # roll over the state to next time step
            print('\rEpisode {}\t Current Score: {:.2f}'.format(episode_cnt, score), end="")
            if done:                                       # exit loop if episode finished
                scores.append(score)
                break
        #if score < 13:
        #    print("\rAgent failed the test! Score: {}".format(score))
        #    break
        #else:
        episode_cnt += 1

    avg_score = np.mean(scores)
    print("\nMean Score: {} after {} episodes".format(avg_score, episode_cnt))

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.axhline(y = avg_score, color = 'r', linestyle = '-', label="my avg score")
    plt.axhline(y = 13, color = 'g', linestyle = 'dashed', label="problem solved" )    
    plt.legend(bbox_to_anchor = (1.0, 1), loc = 'upper center')
    plt.show()

if __name__ == "__main__":
    main()