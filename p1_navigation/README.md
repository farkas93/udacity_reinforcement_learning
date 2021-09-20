[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, I have trained an agent to navigate (and collect bananas!) in a large, square world.  

For watching my agent solving the problem watch the mp4 file called banana_collecting_avg15_76.mp4 which is in this project folder.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of my agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.
My agent got an average score of 16.85 over 100 consecutive episodes during my test.

### Getting Started
1. Download this git repository [click here](https://github.com/farkas93/udacity_reinforcement_learning)

2. Follow the instructions in the root folders README.md (section `Dependencies`) to setup your python environment as needed for this course.

3. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

4. Place the file in the repository you downloaded in step 1, in the `p1_navigation/` folder, and unzip (or decompress) the file. Rename the unzipped folder to `Banana`!

### Instructions

There are multiple ways to run it. You can either use the jupyter notebook file `Navigation.ipynb` or you can use the following two files to run it independently from jupyter:
- `main_train.py` - For training the Deep-QNet.
- `main_test.py` - For testing the Deep-QNet over 100 episodes.

The notebook `Navigation.ipynb` should be self explaining. Nonetheless I recommend using `main_train.py` and `main_test.py`, since I have abandoned `Navigation.ipynb` at some point and therefore the results might differ!

Note that you can pass for both `main_train.py` and `main_test.py` a model name (string) as an argument. The passed model name will be used to create at the end of the training an export pth-file of the agents weights into the folder `saved_models`.
In `main_train.py` it will train a dueling DQNet model for 1600 episodes. Parameters can be changed or set in `dqn_agent.py`.
The agent saves in the `p1_navigation` folder the current weights in a file called `checkpoint.pth` after every epoch (100 episodes per default), iff the model has achieved a better average score than in the best scoring epoch up until then. 
If the agent fails for three epochs to improve the average score, we experience an early out and the agent will be reset to the last checkpoint in which we experienced an improvement. After reset, the training ends and the model will be saved.

For testing the model you can start `main_test.py` with passing the model name (file name in `saved_models` folder without the .pth extension). By default, `main_test.py` will load the weights from `checkpoint.pth` in the project folder. 
Please make sure that the weights you are trying to load indeed match the architecture of the neural network which is defined in `agents/dqn_agent.py`. The available neural network architectures you can find in the `models/linear_models.py`.
