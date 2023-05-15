import numpy as np
import _pickle as pickle
import gym
from gym import wrappers

# Hyperparams
H = 225                     # Neruon count
batch_size = 25             # Used to perform a RMS prop param update every batch_size steps
learning_rate = 1e-3        # Learning rate used in RMS prop
gamma = 0.95               # Discount factor for reward
decay_rate = 0.95          # Decay factor for RMSProp

# Config flags 
resume = False  # Resume training from previous checkpoint (from save.p  file)?
render = True  # Render video output?

# Model initialization
D = 80 * 80  # Input dimensionality: 80x80 grid
if resume:
    model = pickle.load(open('save.p', 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(H, D) / np.sqrt(D)  # "Xavier" initialization - Shape will be H x D
    model['W2'] = np.random.randn(4, H) / np.sqrt(H)  # Shape will be 4 x H

grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}  # Update buffers that add up gradients over a batch
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}  # RMSprop memory

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def prepro(I):
    I = I[34:194, :, :]                 # crop - remove 34px from top and 16px from bottom of image in y, to reduce redundant parts of image (i.e. scoreboard)
    I = I[::2, ::2, 0]                  # downsample by factor of 2
    I[I == 144] = 0                     # erase background (background type 1)
    I[I == 109] = 0                     # erase background (background type 2)
    I[I != 0] = 1                       # everything else (paddles, ball) just set to 1
    return I.astype(float).ravel()


def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def policy_forward(x):
    h = np.dot(model["W1"], x)
    h[h < 0] = 0  # ReLU introduces non-linearity
    logp = np.dot(model["W2"], h)
    p = softmax(logp)
    return p, h


def policy_backward(eph, epx, epdlogp):
    dW2 = np.dot(eph.T, epdlogp)
    dh = np.dot(epdlogp, model["W2"])
    dh[eph <= 0] = 0
    dW1 = np.dot(epx.T, dh)
    return {"W1": dW1.T, "W2": dW2.T}






env = gym.make("Breakout-v0")
observation = env.reset()
preprocessed_observation = prepro(observation)
D = preprocessed_observation.size
prev_x = None
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0

while True:
    if render: env.render()

    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    aprob, h = policy_forward(x)
    action = np.random.choice(4, p=aprob)

    xs.append(x)
    hs.append(h)
    y = np.zeros(4)
    y[action] = 1
    dlogps.append(y - aprob)

    observation, reward, done, info = env.step(action)
    reward_sum += reward
    drs.append(reward)

    if done:
        episode_number += 1
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs, hs, dlogps, drs = [], [], [], []

        discounted_epr = discount_rewards(epr)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= (np.std(discounted_epr) + 1e-8)

        epdlogp *= discounted_epr
        grad = policy_backward(eph, epx, epdlogp)
        for k in model: grad_buffer[k] += grad[k]

        if episode_number % batch_size == 0:
            for k, v in model.items():
                g = grad_buffer[k]
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v)

        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print(f"Resetting env. episode {episode_number} reward total was {reward_sum}. running mean: {running_reward}")
        if episode_number % 100 == 0: pickle.dump(model, open("save.p", "wb"))
        reward_sum = 0
        observation = env.reset()
        prev_x = None
