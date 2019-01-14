import yaml
# import functools as f
# import hashlib


class JobConfig:
    def __init__(self, filename=None):
        if filename:
            self.from_filename(filename)
        else:
            self.random_seed = None
            self.buffer_size = None
            self.batch_size = None
            self.gamma = None
            self.tau = None
            self.lr_actor = None
            self.lr_critic = None
            self.weight_decay = None
            self.sigma = None
            self.actor_nn_size = None
            self.critic_nn_size = None
            self.batch_norm = None
            self.clip_grad_norm = None
            self.n_episodes = None
            self.score_window_size = None
            self.print_every = None
            self.max_score = None
            self.damp_exploration_noise = None
            self.n_episodes = None
            self.score_window_size = None
            self.print_every = None
            self.max_score = None
            self.damp_exploration_noise = None
            self.render_game = None

    def __hash__(self):
        return hash(repr(sorted(self.__dict__.items())))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def from_filename(self, filename):
        "Initialize from a file"
        with open(filename, 'r') as f:
            obj = yaml.load(f)
        self.__dict__ = obj.__dict__

    def dump(self, filename):
        with open(filename, "w") as f:
            yaml.dump(self, f)


class MetaConfig:
    def __init__(self, filename):
        self.from_filename(filename)

    def from_filename(self, filename):
        "Initialize from a file"
        with open(filename, 'r') as f:
            obj = yaml.load(f)
        self.__dict__ = obj.__dict__

    def dump(self, filename):
        with open(filename, "w") as f:
            yaml.dump(self, f)
