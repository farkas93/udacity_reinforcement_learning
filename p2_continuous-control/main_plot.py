
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import pandas as pd


TARGET_SCORE = 30

def plot_scores(scores, episode_solved, rolling_window=100):
    """Plot scores and optional rolling mean using specified window."""
    plt.title("Scores")
    means  = []
    for s in scores:
        means.append(np.mean(s))
    rolling_mean = pd.Series(means).rolling(rolling_window).mean()       
    # plot the scores
    fig = plt.figure()
    plt.plot(np.arange(len(means)), means, label="AVG-Score over all agents / episode")
    plt.plot(rolling_mean, label="AVG-Score over previous 100 episodes")
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.axhline(y = rolling_mean[len(rolling_mean)-1], color = 'r', linestyle = '-', label="AVG-Score last Epoch")
    plt.axhline(y = TARGET_SCORE, color = 'g', linestyle = 'dashed', label="Target Score")   
    
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
    print("Start plotting for the model named: {}".format(model_name))
    
    content = list(csv.reader(open("log_"+model_name+".csv")))
    episode_solved_num = int(content[0][1])
    
    scores = np.array(content[2:][:]).astype(np.float)
    
    plot_scores(scores, episode_solved_num)

if __name__ == "__main__":
    main()