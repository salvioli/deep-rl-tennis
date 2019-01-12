[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"

# deep-rl-tennis
Two deep reinforcement learning agents learning to play tennis in a Unity ML-Agents environment.


## Project details
This project is my solution to the collaboration and competition project of udacity's 
[Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

### Environment and task description
Two agents control a tennis racket each and their goal is to bounce a ball over a net separating them. 
For this project the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment from unity ml agents is used.

![Trained Agent][image1]

- _reward function_: Each of the agents is rewarded as follows:
  - +0.1 if the agent hits the ball over the net
  - -0.01 if the agent lets a ball hit the ground or hits the ball out of bounds
  
- _observation space_: the observation space is continuous and consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.
- _action space_: two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

### Solution criteria
The task is considered solved if the two agents receive at least a collective score of +0.5 over 100 episodes. The collective score is calculated as follows:
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The task is episodic.

## Code description
*TODO*

## Getting Started
### Prerequisites
A working python 3 environment is required. You can easily setup one installing [anaconda] (https://www.anaconda.com/download/)

### Installation and execution
If you are using anaconda is suggested to create a new environment as follows:
```bash
conda create --name tennis python=3.6
```
activate the environment
```bash
source activate tennis
```
start the jupyter server
```bash
python jupyter-notebook --no-browser --ip 127.0.0.1 --port 8888 --port-retries=0
```

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

1. Decompress the archive at your preferred location (e.g. in this repository working copy).
1. Open Report.ipynb notebook 
1. Write your path to the pre-compiled unity environment executable as indicated in the notebook.
1. Follow the instructions in `Tennis.ipynb` to get an environment introduction and to see two random agents playing tennis.
1. **TODO**
