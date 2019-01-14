import numpy as np
from collections import deque
import torch
import matplotlib.pyplot as plt
import time
import os
import pickle
import sys
from unityagents import UnityEnvironment
from ddpg_agent import DDPGAgent
from config import JobConfig, MetaConfig
import random
import calendar


def train(agent, env, n_episodes=1000, score_window_size=100, print_every=50,
          max_score=None, damp_exploration_noise=False):

    temp_save_path = generate_temp_session_path(agent)

    task_solved = False

    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)

    shadow_agent = agent.clone()

    scores_deque = deque(maxlen=score_window_size)
    all_scores = []
    all_avg_scores = []
    all_std = []

    all_episodes_durations = []
    last_max_score = -float('Inf')

    start = time.clock()
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]

        agent.reset()
        shadow_agent.reset()

        states = env_info.vector_observations  # get the current state
        scores = np.zeros(num_agents)  # initialize the score
        while True:

            if damp_exploration_noise:
                damping = (0.5 - np.mean(scores))/0.5 + 0.05
                actions = agent.act(states[0], noise_damping=damping)  # select an action
            else:
                actions = agent.act(states[0])  # select an action

            shadow_actions = shadow_agent.act(states[1], add_noise=False)
            actions = np.array([actions, shadow_actions]).squeeze()

            env_info = env.step(actions)[brain_name]  # send all actions to the environment

            next_states = env_info.vector_observations  # get next state
            rewards = env_info.rewards  # get reward
            dones = env_info.local_done  # see if episode finished

            agent.step(states[0], actions[0], rewards[0], next_states[0], dones[0])

            scores += rewards  # update the score
            states = next_states  # roll over states to next time step

            if np.any(dones):  # exit loop if episode finished
                break

        scores_deque.append(np.max(scores))
        all_scores.append(scores)
        all_avg_scores.append(np.mean(scores_deque))
        all_std.append(np.std(scores_deque))

        end = time.clock()
        all_episodes_durations.append(end-start)
        start = end

        # Every time we improve we update the shadow agent with best weights
        if np.mean(scores_deque) > last_max_score:
            last_max_score = np.mean(scores_deque)
            shadow_agent.copy_weights_from(agent)
            save_state(agent, all_avg_scores, all_scores, all_std,
                       all_episodes_durations, temp_save_path)

        message = '\rEpisode {}\tAverage Score: {:.2f}\tEpisode duration {:.2f}'\
                        .format(i_episode, np.mean(scores_deque),
                        np.mean(all_episodes_durations[-score_window_size:]))

        if np.mean(scores_deque) >= max_score and not task_solved:
            print(f'\nTask solved in {i_episode} episodes\tAverage Score: {np.mean(scores_deque):.2f}')
            task_solved = True
        elif i_episode == n_episodes:
            print('')
        elif i_episode % print_every == 0:
            print(message)
        else:
            print(message, end="")

    save_path = generate_final_session_path(agent, all_avg_scores)
    save_state(agent, all_avg_scores, all_scores, all_std, all_episodes_durations, save_path)

    return all_scores, all_avg_scores, all_std, save_path


def save_state(agent, all_avg_scores, all_scores, all_std, all_episodes_durations, save_path):
    os.makedirs(save_path, exist_ok=True)
    torch.save(agent.actor_local.state_dict(), save_path + 'checkpoint_actor.pth')
    torch.save(agent.critic_local.state_dict(), save_path + 'checkpoint_critic.pth')
    with open(save_path + "scores.p", "wb") as f:
        pickle.dump(all_scores, f)
    with open(save_path + "avg_scores.p", "wb") as f:
        pickle.dump(all_avg_scores, f)
    with open(save_path + "std.p", "wb") as f:
        pickle.dump(all_std,f)
    with open(save_path + "episodes_durations.p", "wb") as f:
        pickle.dump(all_episodes_durations, f)

    plot_scores(all_scores, all_avg_scores, all_std, out_file=save_path + "training_plot.pdf")


def generate_temp_session_path(agent):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    # randomize among different runs
    random.seed(calendar.timegm(time.gmtime()))
    agent_type = agent.__class__.__name__
    folder_name = agent_type + '-' + timestr + '-' + '{:x}'.format(random.randrange(16**30))[:8]
    save_path = './sessions/' + folder_name + '/'
    return save_path


def generate_final_session_path(agent, all_avg_scores):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    agent_type = agent.__class__.__name__
    folder_name = agent_type + '-' + f'{np.max(all_avg_scores):.2f}' + '-' + f'{np.argmax(all_avg_scores)}' + '-' + timestr
    save_path = './checkpoints/' + folder_name + '/'
    return save_path


def plot_scores(scores, avgscores, std, out_file=''):

    if out_file:
        was_interactive = plt.isinteractive()
        plt.ioff()

    scores = np.array(scores)
    avgscores = np.array(avgscores)
    std = np.array(std)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    idxs = np.arange(1, len(avgscores) + 1)

    plt.plot(idxs, scores, alpha=0.3, label=('agent', 'shadow'))
    plt.plot(idxs, avgscores, 'b--')

    min_error = avgscores - 2 * std
    max_error = avgscores + 2 * std

    plt.fill_between(np.arange(1, len(scores) + 1), min_error, max_error, color='blue', alpha=0.1)
    plt.ylabel('Score')
    plt.xlabel('Episode #')

    if not out_file:
        plt.show()
    else:
        plt.savefig(out_file)
        plt.close(fig)
        plt.interactive(was_interactive)


def demo(agent, env, num_episodes=10):
    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=False)[brain_name]
    num_agents = len(env_info.agents)

    states = env_info.vector_observations                  # get the current state
    scores = np.zeros(num_agents)                          # initialize the score
    for i_episode in range(num_episodes):
        actions = agent.act(states, add_noise=False)       # select an action

        env_info = env.step(actions)[brain_name]           # send all actions to the environment
        next_states = env_info.vector_observations         # get next state
        rewards = env_info.rewards                         # get reward
        dones = env_info.local_done                        # see if episode finished

        scores += rewards                                  # update the score
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            continue

    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))


def generate_random_configuration_files(metaconf_file, output_path='./config/'):
    c = MetaConfig(metaconf_file)

    if output_path[-1] != '/':
        raise ValueError

    samples = set()
    while len(samples) < c.n_samples*c.n_random_seeds:
        # generate sampled configuration dictionary
        s = JobConfig()

        for k, v in c.choices.items():
            if isinstance(v, list):
                setattr(s, k, random.choice(v))
            else:
                setattr(s, k, v)

        for k, v in c.continuous.items():
            setattr(s, k, random.uniform(v[0], v[1]))

        for _ in range(c.n_random_seeds):
            setattr(s, 'random_seed', random.randint(1, 100))
            samples.add(s)

    os.makedirs(output_path)
    for x in samples:
        x.dump(output_path + 'config-' + '{:x}'.format(random.randrange(16**30))[:8] + '.yml')

    return output_path


def train_ddpg_agent_job(config):
    if config.render_game:
        env = UnityEnvironment(file_name="./resources/Tennis_Linux/Tennis.x86_64")
    else:
        env = UnityEnvironment(file_name="./resources/Tennis_Linux_NoVis/Tennis.x86_64")

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space
    states = env_info.vector_observations
    print(states)
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:\n', states[0])

    # Train the agent
    agent = DDPGAgent(state_size, action_size,
            config.random_seed,
            config.buffer_size,
            config.batch_size,
            config.gamma,
            config.tau,
            config.lr_actor,
            config.lr_critic,
            config.weight_decay,
            config.sigma,
            config.actor_nn_size,
            config.critic_nn_size,
            config.batch_norm,
            config.clip_grad_norm)

    scores, avg_scores, std, save_path = train(agent, env,
                                               config.n_episodes,
                                               config.score_window_size,
                                               config.print_every,
                                               config.max_score,
                                               config.damp_exploration_noise)

    config.dump(save_path + 'config.yml')

    env.close()

    return scores, avg_scores, std


if __name__ == '__main__':
    idxs = [i for i, s in enumerate(sys.argv) if s.endswith('tennis.py')]
    command = sys.argv[idxs[0]+1]
    args = sys.argv[idxs[0]+2:]
    if command == "train":
        idx = args.index('-f')
        filename = args[idx+1]
        config = JobConfig(filename)
        train_ddpg_agent_job(config)

    elif command == "hyperopt":
        if args[0] == '-f':
            metaconf_file = args[1]
            path = generate_random_configuration_files(metaconf_file)

        else:
            raise (ValueError, f"Unknown parameter{args[0]}")

    else:
        raise(ValueError, f"Unknown command {command}")
