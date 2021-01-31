import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_graph(y1, y2, x1, ENV_NAME, NO_OF_TRAINERS, ROLLING, HOW_MANY_VALUES = None):
    if(HOW_MANY_VALUES == None):
        HOW_MANY_VALUES = len(y1)
    x1 = x1[:HOW_MANY_VALUES]
    y1 = y1[:HOW_MANY_VALUES]
    y2 = y2[:HOW_MANY_VALUES]
    plt.figure(figsize=[12, 9])
    plt.subplot(1, 1, 1)
    plt.title(ENV_NAME)
    plt.xlabel('Steps:')
    plt.ylabel('Avg Reward after 3 runs')
    plt.plot(x1, y1, color='lightgreen')
    plt.plot(x1, y2, color='pink')
    plt.plot(x1, pd.DataFrame(y1)[0].rolling(ROLLING).mean(), color='green')
    plt.plot(x1, pd.DataFrame(y2)[0].rolling(ROLLING).mean(), color='red')
    plt.grid()
    # plt.legend()

    
    plt.savefig('/home/niranth/Desktop/Work/USC_Task/rl_algos/plot_data/Sprites-v0_avg_rewards_enc.png')
    # plt.close()
    plt.show()

if __name__ == "__main__":
    # ENV_NAME = 'Acrobot-v1'
    # ENV_NAME = 'CartPole-v0'
    ENV_NAME = 'Sprites'
    # import pdb; pdb.set_trace()
    # y1 = torch.load('plot_data/Sprites-v0_avg_rewards_enc.pt')
    # y2 = torch.load('plot_data/Sprites-v0_avg_rewards_enc.pt')
    y1 = torch.load('/home/niranth/Desktop/Work/USC_Task/rl_algos/plot_data/Sprites-v0_avg_rewards_enc.pt')
    y2 = torch.load('/home/niranth/Desktop/Work/USC_Task/rl_algos/plot_data/Sprites-v0_avg_rewards_enc.pt')
    # x1 = np.loadtxt('arrays/steps_'+ENV_NAME+'.csv')
    x1 = np.arange(0,len(y1),1)
    plot_graph(y1, y2, x1, ENV_NAME, 3, 50)