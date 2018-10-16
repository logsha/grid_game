#coding=utf-8
"""
Asynchronous Advantage Actor Critic (A3C) with discrete action space, Reinforcement Learning.

The Cartpole example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""
import copy
import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import os
import shutil

OUTPUT_GRAPH = True
LOG_DIR = './log'
N_WORKERS = multiprocessing.cpu_count()
MAX_GLOBAL_EP = 400000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.95
ENTROPY_BETA = 0.002
LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0

GLOBAL_N = 16
GLOBAL_M = 16
N_S = GLOBAL_M * 4 + 2
N_A = GLOBAL_M * GLOBAL_N
EPS = 1e-8
INF = 1e8

def check_valid(s, r, c):
    return s[r] > c
def option(s):
    valid = np.zeros((GLOBAL_M * GLOBAL_N))
    for i in range(GLOBAL_M):
        for j in range(s[i]):
            pos = i * GLOBAL_N + j
            valid[pos] = 1.0
    valid[0] = 0
    return valid

def add_feature(s):
    ls = [s[i] for i in range(GLOBAL_M)]
    # add 块数
    cnt = 0
    for i in range(GLOBAL_M):
        cnt += ls[i]
    ls.append(cnt)
    # add 列数
    cnt = 0
    for i in range(GLOBAL_M):
        if ls[i] > 0:
            cnt += 1
    ls.append(cnt)
    # 最长连续相等块
    cnt, ma = 1, 1
    for i in range(GLOBAL_M):
        if ls[i] == ls[i-1]:
            cnt += 1
        else:
            cnt = 1
        ma = max(ma, cnt)
    ls.append(ma)
    # 与右块是否相同
    for i in range(GLOBAL_M - 1):
        if ls[i] == ls[i+1] and ls[i] > 0:
            ls.append(1)
        else:
            ls.append(0)
    # 与左右构成等差数列
    for i in range(1, GLOBAL_M - 1):
        if ls[i] > 0 and ls[i] * 2 == ls[i+1] + ls[i-1]:
            ls.append(1)
        else:
            ls.append(0)
    # 奇偶性
    cnt = 0
    for i in range(GLOBAL_M):
        ls.append(ls[i]&1)
        cnt += (ls[i]&1)
    ls.append(cnt)
    ls.append(GLOBAL_M - cnt)
    return np.array(ls, dtype=int)

def transform(s):
    return np.array(add_feature(s), dtype=int)
    '''
    if s.ndim < 2: s = s[np.newaxis, :]
    out = []
    for ls in s:
        out.append(add_feature(ls))
    return np.array(out, dtype=int)
    '''

class Cake(object):
    """ n is row limit, m is column limit"""

    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.win_reward = 100.0
        self.lose_reward = -100.0
        self.step_reward = 0

    def read_memory(self, input):
        self.to_learn = []
        self.memory = []
        with open(input, mode='r') as f:
            for line in f.readlines():
                self.to_learn.append([int(i) for i in line.split(',')])
    def add_memory(self):
        siz = len(self.memory)
        if siz < len(self.to_learn):
            self.memory.append(self.to_learn[siz])

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
    def decide(self, r, c):
        for i in range(r, self.m):
            self.board[i] = min(self.board[i], c)

    def rival(self, action_space):
        if self.over():
            return self.lose_reward, self.get_board(), True
        if self.board[1] == 0:
            if self.board[0] == 1:
                self.board[0] = 0
                return self.win_reward, self.get_board(), True
            else:
                self.board[0] = 1
                return self.step_reward, self.get_board(), False
        if self.board[0] == 1:
            for i in range(1, self.m):
                self.board[i] = 0
            return self.step_reward, self.get_board(), False
        valid = option(self.board)
        valid[1] = valid[GLOBAL_N] = 0
        if max(valid) <= 0:
            self.board[0] = 1
            assert self.board[1] == 1 and self.board[2] == 0
            return self.step_reward, self.get_board(), False
        pos = []
        for i in range(len(valid)):
            if valid[i] > 0:
                pos.append((action_space[i], i))
        action = min(pos)[1]
        self.decide(action/GLOBAL_N, action%GLOBAL_N)
        if self.over():
            return self.win_reward, self.get_board(), True
        return self.step_reward, self.get_board(), False            
    def agent_move(self):
        if self.board[1] == 0:
            if self.board[0] == 1:
                self.board[0] = 0
            else:
                self.board[0] = 1
            return
        if self.board[0] == 1:
            for i in range(1, self.m):
                self.board[i] = 0
            return
        if self.board[2] > 0 and self.board[0] == self.board[1] + 1:
            for i in range(2, self.m):
                self.board[i] = 0
            return
        if self.board[2] == 0:
            if self.board[0] > self.board[1] + 1:
                self.board[0] = self.board[1] + 1
                return
            if self.board[0] == self.board[1]:
                self.board[1] = self.board[0] - 1
                return
        '''
        def optimal_move(self, p):
            siz = len(self.memory[p])
            if self.board[:siz] == self.memory[p] and self.board[siz] > 0:
                for i in range(siz, self.m):
                    self.board[i] = 0
                return True
            if self.board[siz] > 0:
                return False
            for i in range(siz):
                if self.board[i] < self.memory[p][i]:
                    return False
                if self.board[i] > self.memory[p][i]:
                    flag = 0
                    for j in range(i, siz):
                        if self.memory[p][j] != min(self.memory[p][i], self.board[j]):
                            flag = 1
                            break
                    if flag == 0:
                        for j in range(siz):
                            self.board[j] = self.memory[p][j]
                        return True
            return False
        for i in range(len(self.memory)):
            if optimal_move(self, i):
                return
        '''
        valid = option(self.board)
        valid[1] = valid[GLOBAL_N] = 0
        pos = []
        for i in range(len(valid)):
            if valid[i] > 0:
                pos.append(i)
        if len(pos) <= 0:
            for i in range(self.m):
                self.board[i] = 0
        else:
            action = pos[np.random.randint(0, len(pos))]
            for i in range(action/GLOBAL_N, self.m):
                self.board[i] = min(self.board[i], action%GLOBAL_N)

    def move(self, r, c):
        if not self.valid(r, c):
            print(self.board)
            print(r, c)
            print("something wrong")
        for i in range(r, self.m):
            self.board[i] = min(self.board[i], c)
        if not self.check():
            print("move error")
        if self.over():
            return self.lose_reward, self.get_board(), True
        self.agent_move();
        if not self.check():
            print("agent error")
        if self.over():
            return self.win_reward, self.get_board(), True
        return self.step_reward, self.get_board(), False


def encode(x, y):
    return x * GLOBAL_N + y
'''
def bad():
    if len(win) > 3 and flag < 10:
        for i in range(3):
            if(win[-1-i] > 100):
                return False
        return True
    return False    
def good():
    if len(win) > 3 and flag < 10:
        if(sum(win[-3:]) > 5000):
                return True
        return False
    return False
'''
def test(ti, ac):
    global GLOBAL_N, GLOBAL_M
    print("testing~~~")
    board = Cake(GLOBAL_N, GLOBAL_M)
    show_ti = 20
    win_time = 0
    for i in range(ti):
        s = transform(board.random_start())
        done = False
        r_sum = 0
        while not done:
            action = ac.choose_action(s)
            #if i < show_ti:
            #    print(s[:GLOBAL_M], action/GLOBAL_N, action%GLOBAL_N)
            reward, s_, done = board.move(action/GLOBAL_N, action%GLOBAL_N)
            s = transform(s_)
            r_sum += reward
        #if i < show_ti:
        #    print("end one episode")
        if r_sum > 0:
            win_time += 1
    print("test done~~~")
    print(win_time)

class ACNet(object):
    def __init__(self, scope, globalAC=None):
        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                self.a_prob, self.v, self.a_params, self.c_params = self._build_net(scope)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(tf.log(self.a_prob + EPS) * tf.one_hot(self.a_his, N_A, dtype=tf.float32), axis=1, keep_dims=True)
                    exp_v = log_prob * td
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + EPS),
                                             axis=1, keep_dims=True)  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0, .1)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 160, tf.nn.relu6, kernel_initializer=w_init, name='la')
            l_b = tf.layers.dense(l_a, 100, tf.nn.relu6, kernel_initializer=w_init, name='lb')
            a_prob = tf.layers.dense(l_b, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 160, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            l_d = tf.layers.dense(l_c, 100, tf.nn.relu6, kernel_initializer=w_init, name='ld')
            v = tf.layers.dense(l_d, 1, kernel_initializer=w_init, name='v')  # state value
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return a_prob, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        # observation = observation[np.newaxis, :]
        n = GLOBAL_N
        valid = option(observation)
        if observation[1] == 0:
            if observation[0] == 1:
                return 0
            else:
                return 1
        if observation[0] == 1:
            return n
        else:
            valid[n] = 0
        valid[1] = 0
        observation = observation[np.newaxis, :]
        # forward feed the observation and get q value for every actions
        actions_value = SESS.run(self.a_prob, feed_dict={self.s: observation})[0]
        # print(valid.shape, actions_value.shape)
        if sum(valid) <= EPS:
            return 1
        #np.clip(actions_value, EPS, 1.0)
        if sum(actions_value) is np.nan:
            print(actions_value)
            print(observation)
            print(valid)
            assert(3==2)
        valid *= actions_value
        if sum(valid) is np.nan:
            print(actions_value)
            print(valid)
            print(observation)
            assert(1==2)
        valid /= sum(valid)
        for i in range(len(valid)):
            if valid[i] < EPS:
                valid[i] = 0
        assert(np.nan not in actions_value)
        assert(np.nan not in valid)
        if sum(valid) is np.nan or sum(valid) <= EPS:
            assert(1==2)
        valid /= sum(valid)
        action = np.random.choice(len(valid), p=valid)
        if not check_valid(observation[0][:GLOBAL_M], action/GLOBAL_N, action%GLOBAL_N):
            print(actions_value)
            print(valid)
            print(observation)
            assert(4==5)
        return action

class Worker(object):
    def __init__(self, name, globalAC):
        global GLOBAL_N, GLOBAL_M
        self.board = Cake(GLOBAL_N, GLOBAL_M)
        self.name = name
        self.AC = ACNet(name, globalAC)

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP, GLOBAL_N
        total_step = 1
        cnt = 0
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = transform(self.board.random_start())
            ep_r = 0
            while True:
                action = self.AC.choose_action(s)
                reward, s_, done = self.board.move(action/GLOBAL_N, action%GLOBAL_N)
                s_ = transform(s_)
                #a = self.AC.choose_action(s)
                #s_, r, done, info = self.env.step(a)
                ep_r += reward
                buffer_s.append(s)
                buffer_a.append(action)
                buffer_r.append(reward)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)

                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * ep_r)
                    cnt += 1
                    if cnt % 1000 == 0 and self.name[-1] == '0':
                        test(2000, self.AC)
                    '''
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                          )
                    '''
                    GLOBAL_EP += 1
                    break

if __name__ == "__main__":
    SESS = tf.Session()

    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(i_name, GLOBAL_AC))

    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, SESS.graph)

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)

