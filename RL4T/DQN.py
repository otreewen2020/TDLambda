# 执行的方式 Python 04_q_learning_for_trading.py > DQN.log

import warnings
warnings.filterwarnings('ignore') # 忽略warning错误信息

from pathlib import Path
from time import time
from collections import deque
from random import sample

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
# from trading_env import  as trading_environment

import gym
from gym.envs.registration import register

np.random.seed(42)
tf.random.set_seed(42)

# 将时间信息格式化
def format_time(t):
    m_, s = divmod(t, 60)
    h, m = divmod(m_, 60)
    return '{:02.0f}:{:02.0f}:{:02.0f}'.format(h, m, s)

# 定义交易天数
trading_days = 252

# 注册环境 进口为 trading_env.py 文件中的TradingEnvironment类
register(
    id='trading-v0',
    entry_point='trading_env:TradingEnvironment', 
    max_episode_steps=trading_days
)

trading_environment = gym.make('trading-v0')
trading_environment.env.trading_days = trading_days
trading_environment.env.trading_cost_bps = 1e-3
trading_environment.env.time_cost_bps = 1e-4
trading_environment.env.ticker = 'AAPL'
trading_environment.seed(42)

# 环境参数
state_dim = trading_environment.observation_space.shape[0] 
num_actions = trading_environment.action_space.n
max_episode_steps = trading_environment.spec.max_episode_steps

print (state_dim) # 观测的维度
print (num_actions) # 动作数量
print (max_episode_steps) # 最大步数

# 定义交易agent
class DDQNAgent:
    def __init__(self, state_dim,
                 num_actions,
                 learning_rate,
                 gamma,
                 epsilon_start,
                 epsilon_end,
                 epsilon_decay_steps,
                 epsilon_exponential_decay,
                 replay_capacity,
                 architecture,
                 l2_reg,
                 tau,
                 batch_size):

        self.state_dim = state_dim                               # 10个数据
        self.num_actions = num_actions                           # 3个动作: long ,hold, short
        self.experience = deque([], maxlen=replay_capacity)      # 用于记录（s, a, s', r, done）
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.architecture = architecture
        self.l2_reg = l2_reg

        self.online_network = self.build_model()                 
        self.target_network = self.build_model(trainable=False)
        self.update_target()

        self.epsilon = epsilon_start
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.epsilon_exponential_decay = epsilon_exponential_decay
        self.epsilon_history = []

        self.total_steps = self.train_steps = 0
        self.episodes = self.episode_length = self.train_episodes = 0
        self.steps_per_episode = []
        self.episode_reward = 0
        self.rewards_history = []

        self.batch_size = batch_size
        self.tau = tau
        self.losses = []
        self.idx = tf.range(batch_size)
        self.train = True

    def build_model(self, trainable=True):
        layers = []
        n = len(self.architecture)
        for i, units in enumerate(self.architecture, 1):
            layers.append(Dense(units=units,
                                input_dim=self.state_dim if i == 1 else None,
                                activation='relu',
                                kernel_regularizer=l2(self.l2_reg),
                                name=f'Dense_{i}',
                                trainable=trainable))
        layers.append(Dropout(.1))
        layers.append(Dense(units=self.num_actions,
                            trainable=trainable,
                            name='Output'))
        model = Sequential(layers)
        model.compile(loss='mean_squared_error',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target(self):
        self.target_network.set_weights(self.online_network.get_weights()) # 将online网络的权重赋值给target网络

    def epsilon_greedy_policy(self, state):
        self.total_steps += 1
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.num_actions)  # 以epsilon的概率随机的选择动作
        q = self.online_network.predict(state)         # 以1-epsilon的概率选择使 Q(s,a)最大的action
        return np.argmax(q, axis=1).squeeze()

    def memorize_transition(self, s, a, r, s_prime, not_done):
        if not_done:
            self.episode_reward += r
            self.episode_length += 1
        else:
            if self.train: # epsilon不断递减
                if self.episodes < self.epsilon_decay_steps:
                    self.epsilon -= self.epsilon_decay
                else:
                    self.epsilon *= self.epsilon_exponential_decay

            self.episodes += 1
            self.rewards_history.append(self.episode_reward)
            self.steps_per_episode.append(self.episode_length)
            self.episode_reward, self.episode_length = 0, 0

        self.experience.append((s, a, r, s_prime, not_done)) # 将(s, a, s_prime, not_done) 写入experience

    def experience_replay(self):
        if self.batch_size > len(self.experience):
            return
        minibatch = map(np.array, zip(*sample(self.experience, self.batch_size)))
        states, actions, rewards, next_states, not_done = minibatch

        next_q_values = self.online_network.predict_on_batch(next_states)
        best_actions = tf.argmax(next_q_values, axis=1)

        next_q_values_target = self.target_network.predict_on_batch(next_states)
        target_q_values = tf.gather_nd(next_q_values_target,
                                       tf.stack((self.idx, tf.cast(best_actions, tf.int32)), axis=1))

        targets = rewards + not_done * self.gamma * target_q_values

        q_values = self.online_network.predict_on_batch(states)
        q_values[[self.idx, actions]] = targets

        loss = self.online_network.train_on_batch(x=states, y=q_values)
        self.losses.append(loss)

        if self.total_steps % self.tau == 0:
            self.update_target()

# 定义模型参数与网络结构
gamma = .99,  # discount factor
tau = 100  # target network update frequency

architecture = (256, 256)  # units per layer
learning_rate = 0.0001  # learning rate
l2_reg = 1e-6  # L2 regularization
replay_capacity = int(1e6)
batch_size = 4096

epsilon_start = 1.0
epsilon_end = .01
epsilon_decay_steps = 250
epsilon_exponential_decay = .99

tf.keras.backend.clear_session()

ddqn = DDQNAgent(state_dim=state_dim,
                 num_actions=num_actions,
                 learning_rate=learning_rate,
                 gamma=gamma,
                 epsilon_start=epsilon_start,
                 epsilon_end=epsilon_end,
                 epsilon_decay_steps=epsilon_decay_steps,
                 epsilon_exponential_decay=epsilon_exponential_decay,
                 replay_capacity=replay_capacity,
                 architecture=architecture,
                 l2_reg=l2_reg,
                 tau=tau,
                 batch_size=batch_size)

ddqn.online_network.summary()


#设置训练参数
total_steps = 0
max_episodes = 1000
# 记录训练数据
episode_time, navs, market_navs, diffs, episode_eps = [], [], [], [], []


def track_results(episode, nav_ma_100, nav_ma_10,
                  market_nav_100, market_nav_10,
                  win_ratio, total, epsilon):
    time_ma = np.mean([episode_time[-100:]])
    T = np.sum(episode_time)
    
    template = '{:>4d} | {} | Agent: {:>6.1%} ({:>6.1%}) | '
    template += 'Market: {:>6.1%} ({:>6.1%}) | '
    template += 'Wins: {:>5.1%} | eps: {:>6.3f}'
    print(template.format(episode, format_time(total), 
                          nav_ma_100-1, nav_ma_10-1, 
                          market_nav_100-1, market_nav_10-1, 
                          win_ratio, epsilon))


# 开始训练
start = time()

results = [] #记录训练结果

for episode in range(1, max_episodes + 1):
    # 将环境重置
    this_state = trading_environment.reset()
    # 每次从step1走到step252
    for episode_step in range(max_episode_steps):
        
        action = ddqn.epsilon_greedy_policy(this_state.reshape(-1, state_dim))
        next_state, reward, done, _ = trading_environment.step(action)
        print("epi:", episode, "|step:", episode_step, "|action:", action, "|next_state:",next_state, "|reward:", reward, "done:", done)

        ddqn.memorize_transition(this_state, 
                                 action, 
                                 reward, 
                                 next_state, 
                                 0.0 if done else 1.0)
        if ddqn.train:
            ddqn.experience_replay()
        if done:
            break
        this_state = next_state
    
    # 用于记录每一步的训练结果 
    result = trading_environment.env.simulator.result()
    # 最后一步的训练结果
    final = result.iloc[-1]

    # 记录策略的NAV
    nav = final.nav * (1 + final.strategy_return)
    navs.append(nav)

     # 记录市场的NAV
    market_nav = final.market_nav
    market_navs.append(market_nav)
    
    # 计算策略与市场的差异
    diff = nav - market_nav
    diffs.append(diff)
    # 每隔10次打印训练结果
    if episode % 10 == 0:
        track_results(episode, np.mean(navs[-100:]), np.mean(navs[-10:]), 
                      np.mean(market_navs[-100:]), np.mean(market_navs[-10:]), 
                      np.sum([s > 0 for s in diffs[-100:]])/min(len(diffs), 100), 
                      time() - start, ddqn.epsilon)
    # 当策略优于市场时训练结束
    if len(diffs) > 25 and all([r > 0 for r in diffs[-25:]]):
        print(result.tail())
        break

trading_environment.close()


# 存储训练结果
results = pd.DataFrame({'Episode': list(range(1, episode+1)),
                        'Agent': navs,
                        'Market': market_navs,
                        'Difference': diffs}).set_index('Episode')

results['Strategy Wins (%)'] = (results.Difference > 0).rolling(100).sum()
results.info()
results.to_csv(results_path / 'dqn_results.csv', index=False)
