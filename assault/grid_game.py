import numpy as np
import matplotlib.pyplot as plt
import dist
import util
import math as m
import random


class Grid_Game:
    ax = None
    ims = None

    def __init__(self, grid_size, stride_factor=1, random_start=True):
        self.q = None
        self.n = grid_size
        self.actions = ['up', 'down', 'left', 'right']
        self.discount_factor = 1
        self.stride = stride_factor
        self.states = [((px,py), (rx,ry)) for px in range(self.n) \
                        for py in range(self.n)
                        for rx in range(self.n)
                        for ry in range(self.n)]
        self.states.append('over')
        if random_start:
            self.start = dist.uniform_dist([((0, 0), (int(self.n/2), ry)) for ry in range(self.n)])
        else:
            self.start = dist.delta_dist(((0, 0), (int(self.n/2), int(self.n/2))))

    def init_state(self):
        return self.start.draw()

    def update_line(num, data, line):
        line.set_data(data[..., :num])
        return line

    def draw_state(self, state=None, pause=False): # From 6.036  course material
        def _update(self, state, pause):
            if self.ax is None:
                plt.ion()
                plt.figure(facecolor="white")
                self.ax = plt.subplot()

            if state is None: state = self.state
            ((px, py), (rx, ry)) = state
            im = np.zeros((self.n, self.n))
            im[px, py] = -1
            im[rx, ry] = 1
            self.ax.cla()
            self.ims = self.ax.imshow(im, interpolation = 'none',
                                    cmap = 'viridis',
                                    extent = [-0.5, self.n+0.5,
                                                -0.5, self.n-0.5],
                                    animated = True)
            self.ims.set_clim(-1, 1)
            plt.pause(0.0001)
            if pause: input('go?')
            else: plt.pause(0.01)
        _update(self, state, pause)


    def state2vec(self, s):
        if s == 'over':
            return np.array([[0, 0, 0, 0, 1]])
        else:
            ((px, py), (rx, ry)) = s
            return np.array([[px, py, rx, ry, 0]])

    def terminal(self, s):
        return s == 'over'

    def reward_fn(self, s, a):
        if s == 'over':
            return 0
        ((px, py), (rx, ry)) = s
        distc = m.sqrt((px-rx)**2 + (py-ry)**2)
        if distc < self.n/3:
            return 1
        else:
            return -1

    def transition_model(self, s, a):

        if s == 'over':
            return dist.delta_dist('over')
        # the state
        ((px, py), (rx, ry)) = s
        # all possible actions
        if a == 'up':
            new_px = px
            if py + 2 > self.n - 1:
                new_py = self.n - 1
            else:
                new_py = py + 2
        if a == 'down':
            new_px = px
            if py - 2 < 0:
                new_py = 0
            else:
                new_py = py - 2
        if a == 'left':
            new_py = py
            if px - 2 < 0:
                new_px = 0
            else:
                new_px = px - 2
        if a == 'right':
            new_py = py
            if px + 2 > self.n - 1:
                new_px = self.n - 1
            else:
                new_px = px + 2
        # end all possible actions

        # movement of reward (rx, ry)
        new_rx_up = rx
        if ry + self.stride > self.n - 1:
            new_ry_up = self.n - 1
        else:
            new_ry_up = ry + self.stride
        new_rx_down = rx
        if ry - self.stride < 0:
            new_ry_down = 0
        else:
            new_ry_down = ry - self.stride
        new_ry_left = ry
        if rx - self.stride < 0:
            new_rx_left = 0
        else:
            new_rx_left = rx - self.stride
        new_ry_right = ry
        if rx + self.stride > self.n - 1:
            new_rx_right = self.n - 1
        else:
            new_rx_right = rx + self.stride
        new_s_up = ((new_px, new_py), (new_rx_up, new_ry_up))
        new_s_down = ((new_px, new_py), (new_rx_down, new_ry_down))
        new_s_left = ((new_px, new_py), (new_rx_left, new_ry_left))
        new_s_right = ((new_px, new_py), (new_rx_right, new_ry_right))
        n_set = set([new_s_up, new_s_down, new_s_left, new_s_right])
        ret_list = list(n_set)
        if rx == new_px:
            return dist.delta_dist('over')
        else:
            return dist.uniform_dist(ret_list)

    def sim_transition(self, s, a):
        return (self.reward_fn(s, a),
                self.init_state() if self.terminal(s) else
                    self.transition_model(s, a).draw())





