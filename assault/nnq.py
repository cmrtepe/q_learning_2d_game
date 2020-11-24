import torch
import torch.nn as nn
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import animation, rc
matplotlib.use("Agg")
import random
from dist import DDist

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_neural_net(state_dims, num_hidden_layers, num_units): # Based on:  Code provided for MIT 6.036 Intro.
                                                               # to Machine Learning HW10

    model = []
    model += [nn.Linear(state_dims, num_units), nn.LeakyReLU()]
    for i in range(num_hidden_layers-1):
        model += [nn.Linear(num_units, num_units), nn.LeakyReLU()]
    model += [nn.Linear(num_units, 1)]
    model = nn.Sequential(*model)

    def init_weights(w):
        if type(w) == nn.Linear: nn.init.xavier_normal_(w.weight)

    return model

class NNQ: # Based on: Code provided for MIT 6.036 Intro. to Machine Learning HW10
    def __init__(self, states, actions, state2vec, num_layers, num_units,
                 lr=1e-2, epochs=1):
        self.running_loss = 0. # To keep a running average of the loss
        self.running_one = 0. # idem
        self.num_running = 0.001 # idem
        self.lr = lr
        self.actions = actions
        self.states = states
        self.state2vec = state2vec
        self.epochs = epochs
        state_dims = state2vec(states[0]).shape[1] # a row vector
        self.models = {a: make_neural_net(state_dims, num_layers, num_units) for a in actions}
    def predict(self, model, s):
        return model(torch.FloatTensor(self.state2vec(s))).detach().numpy()
    def get(self, s, a):
        md = self.models[a]
        return self.predict(md, s)


    def fit(self, model, X,Y, epochs=None, dbg=None): #Code provided for MIT 6.036 Intro. to Machine Learning HW10
        # This function receives two numpy arrays (with shape (K,7) and (K,1)), not two lists!
        assert type(X) is not type([])
        if epochs is None:
            epochs = self.epochs
        train = torch.utils.data.TensorDataset(torch.FloatTensor(X), torch.FloatTensor(Y))
        load = torch.utils.data.DataLoader(train, batch_size=256,shuffle=True)
        opt = torch.optim.SGD(model.parameters(), lr=self.lr)
        for epoch in range(epochs):
            for (X,Y) in load:
                opt.zero_grad()
                loss = torch.nn.MSELoss()(model(X), Y)
                loss.backward()
                self.running_loss = self.running_loss*(1.-self.num_running) + loss.item()*self.num_running
                self.running_one = self.running_one*(1.-self.num_running) + self.num_running
                opt.step()
        if dbg is True or (dbg is None and np.random.rand()< (0.001*X.shape[0])):
            print('Loss running average: ', self.running_loss/self.running_one)

    def update(self, data, lr, epochs = 1):
        for action in self.actions:
          X = np.array([[None]])
          Y = np.array([[None]])
          for s, a, t in data:
            if a == action:
              if np.ndim(t) == 0:
                tt = np.array([[t]])
              elif np.ndim(t) > 2: print("dim error")
              else: tt = t
              if X.any() == None:
                X = self.state2vec(s)
                Y = tt
              else:
                X = np.vstack((X,self.state2vec(s)))
                Y = np.vstack((Y,tt))
          model = self.models[action]
          if not (X.any() == None):
            self.fit(model, X, Y, epochs=epochs)




def sim_episode(mdp, episode_length, policy): # Function from MIT 6.036 Intro. to Machine Learning HW10


    episode = []
    reward = 0
    s = mdp.init_state()
    all_states = [s]
    for i in range(episode_length):
        a = policy(s)
        (r, s_new) = mdp.sim_transition(s, a)
        reward += r
        if mdp.terminal(s):
            episode.append((s, a, r, None))
            break
        mdp.draw_state(s)
        episode.append((s, a, r, s_new))
        s = s_new
        all_states.append(s)
        anime = animate(all_states, mdp.n, episode_length)
    return reward, episode, anime

def animate(states, n, episode_len): # Function from MIT 6.036 Intro. to Machine Learning HW10
    plt.ion()
    plt.figure(facecolor="white")
    fig, ax = plt.subplots()
    plt.close()

    def animate(i):
        if states[i % len(states)] == None or states[i % len(states)] == 'over':
            return
        ((px, py), (rx, ry)) = states[i % len(states)]
        im = np.zeros((n, n))
        im[px, py] = -1
        im[rx, ry] = 1
        ax.cla()
        ims = ax.imshow(im, interpolation='none',
                        cmap='viridis',
                        extent=[-0.5, n + 0.5,
                                -0.5, n - 0.5],
                        animated=True)
        ims.set_clim(-1, 1)
    rc('animation', html='jshtml')
    anim = animation.FuncAnimation(fig, animate, frames=episode_len, interval=100)
    return anim


def Q_learn_batch(mdp, q, lr=.1, iters=100, eps=0.5,
                  episode_length=10, n_episodes=2,
                  interactive_fn=None):
    it = 0
    all_experiences = []
    while it < iters:
        if interactive_fn: interactive_fn(q, it)
        it = it + 1
        policy = lambda s: epsilon_greedy(q, s, eps)
        for i in range(n_episodes):
            reward, episode, animation = sim_episode(mdp, episode_length, policy)
            for ep_tup in episode:
                all_experiences.append(ep_tup)
        all_q_targets = []
        for tup in all_experiences:
            s, a, r, s_new = tup
            gamma = mdp.discount_factor
            if mdp.terminal(s):
                t = r
            else:
                t = r + gamma * value(q, s_new)
            all_q_targets.append((s, a, t))
        q.update(all_q_targets, lr)

    return q


def value(q, s):
    return max(q.get(s, a) for a in q.actions)

def value_iteration(mdp, q, eps = 0.01, max_iters = 1000):
    t = 0
    q_iter = q.copy()
    while t < max_iters:
        t += 1
        new_q = q_iter.copy()
        for s in q.states:
            for a in q.actions:
                f = lambda s: value(q_iter, s)
                v = mdp.reward_fn(s, a) + \
                    mdp.discount_factor * mdp.transition_model(s, a).expectation(f)
                new_q.set(s, a, v)
        input_list = [(p1, p2) for p1 in q.states for p2 in q.actions]
        value_list = [abs(new_q.get(p[0], p[1]) - q_iter.get(p[0], p[1])) for p in input_list]
        max_val = max(value_list)
        if max_val < eps:
            return new_q
        q_iter = new_q
    return q_iter

def greedy(q, s):
    cur = q.actions[0]
    for a in q.actions:
        if q.get(s, a) > q.get(s, cur):
            cur = a
    return cur


def epsilon_greedy(q, s, eps = 0.5):
    n = len(q.actions)
    values = [1 / n] * n
    d = dict(zip(q.actions, values))
    dist = DDist(d)
    if random.random() < eps:  # True with prob eps, random action
        return dist.draw()
    else:
        return greedy(q, s)


