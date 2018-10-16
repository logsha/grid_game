"""
A simple version of Proximal Policy Optimization (PPO) using single thread.
Based on:
1. Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]
2. Proximal Policy Optimization Algorithms (OpenAI): [https://arxiv.org/abs/1707.06347]
View more on my tutorial website: https://morvanzhou.github.io/tutorials
Dependencies:
tensorflow r1.2
gym 0.9.2
"""
import copy
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

GLOBAL_N = 16
GLOBAL_M = 16
EP_MAX = 300000
EP_LEN = 200
GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.0002
BATCH = 32
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM, A_DIM = GLOBAL_M, 1
EPS = 1e-6
INF = 1e-8
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization



class Cake(object):
    """ n is row limit, m is column limit"""

    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.win_reward = 100.0
        self.lose_reward = -100.0
        self.step_reward = -1.0

    def add_feature(self, s):
        ls = [s[i] for i in range(self.m)]
        # add 块数
        cnt = 0
        for i in range(self.m):
            cnt += ls[i]
        ls.append(cnt)
        # add 列数
        cnt = 0
        for i in range(self.m):
            if ls[i] > 0:
                cnt += 1
        ls.append(cnt)
        # 最长连续相等块
        cnt, ma = 1, 1
        for i in range(1, self.m):
            if ls[i] == ls[i-1]:
                cnt += 1
            else:
                cnt = 1
            ma = max(ma, cnt)
        ls.append(ma)
        # 与右块是否相同
        for i in range(self.m - 1):
            if ls[i] == ls[i+1] and ls[i] > 0:
                ls.append(1)
            else:
                ls.append(0)
        # 与左右构成等差数列
        for i in range(1, self.m - 1):
            if ls[i] > 0 and ls[i] * 2 == ls[i+1] + ls[i-1]:
                ls.append(1)
            else:
                ls.append(0)
        # 奇偶性
        cnt = 0
        for i in range(self.m):
            ls.append(ls[i]&1)
            cnt += (ls[i]&1)
        ls.append(cnt)
        ls.append(self.m - cnt)

        return np.array(ls, dtype=int)
    def check(self):
        for i in range(1, self.m):
            if(self.board[i] > self.board[i-1]):
                return False
        return True
    def get_board(self):
        return np.array(self.board, dtype=int)
        #return copy.deepcopy(self.board)

    def start(self):
        self.board = [self.n for i in range(self.m)]
        return self.get_board()

    def random_start(self):
        self.board = [np.random.randint(1, self.n) for i in range(self.m)]
        self.board.sort()
        self.board.reverse()
        return self.get_board()

    def valid(self, r, c):
        return self.board[r] > c and self.check()

    def over(self):
        return self.board[0] == 0

    def agent_move(self):
        if self.board[1] == 0:
            if self.board[0] == 1:
                self.board[0] = 0
            else:
                self.board[0] = 1
            return
        total = 0
        for i in range(self.m):
            total += self.board[i]
        pos = np.random.randint(0, total)
        total = 0
        for i in range(self.m):
            total += self.board[i]
            if total > pos:
                down = self.board[i] - (total - pos)
                for j in range(i, self.m):
                    self.board[j] = min(self.board[j], down)
                return

    def move(self, r, c):
        #if not self.valid(r, c):
        #    print("something wrong")
        for i in range(r, self.m):
            self.board[i] = min(self.board[i], c)
        if not self.check():
            print("move error")
        if self.over():
            return self.win_reward, self.get_board(), True
        self.agent_move();
        if not self.check():
            print("agent error")
        if self.over():
            return self.lose_reward, self.get_board(), True
        return self.step_reward, self.get_board(), False

class PPO(object):

    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # critic
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs, 80, tf.nn.relu, trainable=True)
            l2 = tf.layers.dense(l1, 40, tf.nn.relu, trainable=True)
            self.v = tf.layers.dense(l2, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = pi
        # self.sample_op = tf.squeeze(pi.sample(1), axis=0)       # choosing action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.int32, [None], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                # print(pi.dtype)
                # print(self.tfa.dtype)
                prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pi, labels=self.tfa)
                old_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=oldpi, labels=self.tfa)
                ratio = prob / (EPS + old_prob)
                # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                # ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
                surr = ratio * self.tfadv
            if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else:   # clipping method, find this is better
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv))

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        tf.summary.FileWriter("log/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def update(self, s, a, r):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.tfs: s, self.tfa: a.astype(int32), self.tfadv: adv, self.tflam: METHOD['lam']})
                if kl > 4*METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)    # sometimes explode, this clipping is my solution
        else:   # clipping method, find this is better (OpenAI's paper)
            [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 80, tf.nn.relu, trainable=trainable)
            l2 = tf.layers.dense(l1, 40, tf.nn.relu, trainable=trainable)
            all_act = tf.layers.dense(l2, GLOBAL_N * GLOBAL_M, tf.nn.relu, trainable=trainable)
            # print(all_act.name)
            # act_pro = tf.nn.softmax(all_act, name='act_pro')
            '''
            mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
            '''
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return all_act, params
        
    def choose_action(self, s):
        # if s.ndim < 2: s = s[np.newaxis, :]
        if s[1] == 0:
            if s[0] == 1:
                return 0, 0
            else:
                return 0, 1
        prob_weights = self.sess.run(self.sample_op, {self.tfs: s[np.newaxis, :]})
        valid = np.zeros((GLOBAL_M * GLOBAL_N))
        for i in range(GLOBAL_M):
            for j in range(s[i]):
                pos = i * GLOBAL_N + j
                valid[pos] = np.exp(min(prob_weights[0][pos], INF))
        # cv = copy.deepcopy(valid)
        valid /= valid.sum()
        action = np.random.choice(valid.size, 1, replace=False, p=valid)[0]  # select action w.r.t the actions prob
        return action / GLOBAL_N, action % GLOBAL_N

    '''
    def choose_action(self, s):
        total = 0
        for i in range(GLOBAL_M):
            total += s[i]
        pos = np.random.randint(0, total)
        total = 0
        for i in range(GLOBAL_M):
            total += s[i]
            if total > pos:
                down = s[i] - (total - pos)
                return i, down
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -2, 2)
        '''
    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

BLOCK = 2000
# env = gym.make('Pendulum-v0').unwrapped
board = Cake(GLOBAL_N, GLOBAL_M)
ppo = PPO()
all_ep_r = []

def encode(x, y):
    return x * GLOBAL_N + y

r_sum = 0.0
for ep in range(EP_MAX):
    # s = env.reset()
    s = board.random_start()
    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0
    done = False
    step = 0
    while not done:    # in one episode
        # env.render()
        ax, ay = ppo.choose_action(s)
        r, s_, done = board.move(ax, ay)
        buffer_s.append(s)
        buffer_a.append(encode(ax, ay))
        buffer_r.append(r)    # normalize reward, find to be useful
        s = s_
        ep_r += r
        step += 1
        # update ppo
        if step % BATCH == 0 or done:
            v_s_ = ppo.get_v(s_)
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()
            br = np.array(discounted_r, dtype=float)
            if br.std() > EPS:
                br = (br - br.mean())/br.std()
            bs, ba = np.vstack(buffer_s), np.array(buffer_a, dtype=int)
            buffer_s, buffer_a, buffer_r = [], [], []
            ppo.update(bs, ba, br[:, np.newaxis])
    r_sum += ep_r
    if ep % BLOCK == BLOCK - 1:
        print(r_sum)
        all_ep_r.append(r_sum)
        r_sum = 0.0
        plt.plot(np.arange(len(all_ep_r)), all_ep_r)
        plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward')
        plt.savefig("haha.jpg")
    '''if ep == 0: all_ep_r.append(ep_r)
    else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
    print(
        'Ep: %i' % ep,
        "|Ep_r: %i" % ep_r,
        ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
    )'''

