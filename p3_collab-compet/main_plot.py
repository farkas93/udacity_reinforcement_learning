
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import pandas as pd


TARGET_SCORE = 0.5

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
    avg_score_last_epoch = rolling_mean[len(rolling_mean)-1]
    plt.axhline(y = avg_score_last_epoch, color = 'r', linestyle = '-', label="AVG-Score last Epoch ({:.4f})".format(avg_score_last_epoch))
    plt.axhline(y = TARGET_SCORE, color = 'g', linestyle = 'dashed', label="Target Score")   
    
    plt.axvline(x = episode_solved, color = 'magenta', linestyle = '-', label="Episode {} Solved the Environment".format(episode_solved))    
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