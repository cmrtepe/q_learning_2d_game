
from grid_game import Grid_Game
from nnq import NNQ, value_iteration, Q_learn_batch, greedy, sim_episode


def test_learn_play(d = 10, num_layers = 2, num_units = 100,
                    eps = 0.5, iters = 50, batch_epochs=10,
                    num_episodes = 10, episode_length = 100, nnq_lr=3e-3):
    iters_per_value = 1 if iters <= 10 else int(iters / 10.0) # Based on: test_learn_play test function in
                                                              # MIT 6.036 HW10.
    scores = []
    def interact(q, iter=0):
        if iter % iters_per_value == 0:
            scores.append((iter, evaluate(game, num_episodes, episode_length,
                                          lambda s: greedy(q, s))[0]))
            print('score', scores[-1], flush=True)
    game = Grid_Game(d, stride_factor=1, random_start=True)
    q = NNQ(game.states, game.actions, game.state2vec, num_layers, num_units,
                epochs=batch_epochs, lr=nnq_lr)
    qf = Q_learn_batch(game, q, iters=iters, episode_length=episode_length, n_episodes=num_episodes,
                           interactive_fn=interact)


    for i in range(num_episodes):
        reward, _, animation = sim_episode(game, (episode_length if d > 3 else episode_length/2),
                                lambda s: greedy(qf, s))
        print('Reward', reward)
    return animation



def evaluate(mdp, n_episodes, episode_length, policy):
    score = 0
    length = 0
    for i in range(n_episodes):
        # Accumulate the episode rewards
        r, e, _ = sim_episode(mdp, episode_length, policy)
        score += r
        length += len(e)
        # print('    ', r, len(e))
    return score/n_episodes, length/n_episodes