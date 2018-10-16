#coding=utf-8
import copy
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from dqn import DeepQNetwork
import argparse
import time
GLOBAL_N = 16
GLOBAL_M = 16
EP_MAX = 5000000
BATCH = 16
S_DIM, A_DIM = GLOBAL_M * 4 + 2, GLOBAL_M * GLOBAL_N
EPS = 1e-6
INF = 1e8
MODE = "random"

def option(s):
	valid = np.zeros((GLOBAL_M * GLOBAL_N))
	for i in range(GLOBAL_M):
		for j in range(s[i]):
			pos = i * GLOBAL_N + j
			valid[pos] = 1.0
	return valid

def decide(state, r, c):
	s = state[0].copy()
	for i in range(r, GLOBAL_M):
		s[i] = min(s[i], c)
	return (s, state[1] + 1)
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
	assert(len(s) == GLOBAL_M)
	return np.array(add_feature(s), dtype=int)
	'''
	if s.ndim < 2: s = s[np.newaxis, :]
	'''

class TreeNode(object):
    """A node in the MCTS tree. Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        # This value for u will be overwritten in the first call to update(), but is useful for
        # choosing the first action from this node.
        self._u = prior_p
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children.
        Arguments:
        action_priors -- output from policy function - a list of tuples of actions and their prior
            probability according to the policy function.
        Returns:
        None
        """
        for action, prob in action_priors.iteritems():
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self):
        """Select action among children that gives maximum action value, Q plus bonus u(P).
        Returns:
        A tuple of (action, next_node)
        """
        return max(self._children.iteritems(), key=lambda act_node: act_node[1].get_value())

    def update(self, leaf_value, c_puct):
        """Update node values from leaf evaluation.
        Arguments:
        leaf_value -- the value of subtree evaluation from the current player's perspective.
        c_puct -- a number in (0, inf) controlling the relative impact of values, Q, and
            prior probability, P, on this node's score.
        Returns:
        None
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += (leaf_value - self._Q) / self._n_visits
        # Update u, the prior weighted by an exploration hyperparameter c_puct and the number of
        # visits. Note that u is not normalized to be a distribution.
        if not self.is_root():
            self._u = c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)

    def update_recursive(self, leaf_value, c_puct):
        """Like a call to update(), but applied recursively for all ancestors.
        Note: it is important that this happens from the root downward so that 'parent' visit
        counts are correct.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(leaf_value, c_puct)
        self.update(leaf_value, c_puct)

    def get_value(self):
        """Calculate and return the value for this node: a combination of leaf evaluations, Q, and
        this node's prior adjusted for its visit count, u
        """
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """A simple (and slow) single-threaded implementation of Monte Carlo Tree Search.
    Search works by exploring moves randomly according to the given policy up to a certain
    depth, which is relatively small given the search space. "Leaves" at this depth are assigned a
    value comprising a weighted combination of (1) the value function evaluated at that leaf, and
    (2) the result of finishing the game from that leaf according to the 'rollout' policy. The
    probability of revisiting a node changes over the course of the many playouts according to its
    estimated value. Ultimately the most visited node is returned as the next action, not the most
    valued node.
    The term "playout" refers to a single search from the root, whereas "rollout" refers to the
    fast evaluation from leaf nodes to the end of the game.
    """

    def __init__(self, policy_fn, rollout_policy_fn, c_puct=1.3,
                 rollout_limit=300, playout_depth=10, n_playout=700):
        """Arguments:
        value_fn -- a function that takes in a state and ouputs a score in [-1, 1], i.e. the
            expected value of the end game score from the current player's perspective.
        policy_fn -- a function that takes in a state and outputs a list of (action, probability)
            tuples for the current player.
        rollout_policy_fn -- a coarse, fast version of policy_fn used in the rollout phase.
        lmbda -- controls the relative weight of the value network and fast rollout policy result
            in determining the value of a leaf node. lmbda must be in [0, 1], where 0 means use only
            the value network and 1 means use only the result from the rollout.
        c_puct -- a number in (0, inf) that controls how quickly exploration converges to the
            maximum-value policy, where a higher value means relying on the prior more, and
            should be used only in conjunction with a large value for n_playout.
        """
        self._root = TreeNode(None, 1.0)
        # self._value = value_fn
        self._policy = policy_fn
        self._rollout = rollout_policy_fn

        # self._lmbda = lmbda
        self._c_puct = c_puct
        self._rollout_limit = rollout_limit
        self._L = playout_depth
        self._n_playout = n_playout

    def _playout(self, state, leaf_depth):
        """Run a single playout from the root to the given depth, getting a value at the leaf and
        propagating it back through its parents. State is modified in-place, so a copy must be
        provided.
        Arguments:
        state -- a copy of the state.
        leaf_depth -- after this many moves, leaves are evaluated.
        Returns:
        None
        """
        node = self._root
        for i in range(leaf_depth):
            # Only expand node if it has not already been done. Existing nodes already know their
            # prior.
            if node.is_leaf():
            	# print(state)
                action_probs = self._policy(state[0])
                # print(type(action_probs), action_probs)
                # Check for end of game.
                if len(action_probs) == 0:
                    break
                node.expand(action_probs)
            # Greedily select next move.
            action, node = node.select()
            state = decide(state, action/GLOBAL_N, action%GLOBAL_N)

        # Evaluate the leaf using a weighted combination of the value network, v, and the game's
        # winner, z, according to the rollout policy. If lmbda is equal to 0 or 1, only one of
        # these contributes and the other may be skipped. Both v and z are from the perspective
        # of the current player (+1 is good, -1 is bad).
        # v = self._value(state) if self._lmbda < 1 else 0
        z = self._evaluate_rollout(state, self._rollout_limit)
        # leaf_value = (1 - self._lmbda) * v + self._lmbda * z

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(z, self._c_puct)

    def _evaluate_rollout(self, state, limit):
        """Use the rollout policy to play until the end of the game, returning +1 if the current
        player wins, -1 if the opponent wins, and 0 if it is a tie.
        """
        for i in range(limit):
            action = self._rollout(state[0])
            if action < 0:
                break
            state = decide(state, action/GLOBAL_N, action%GLOBAL_N)
        else:
            # If no break from the loop, issue a warning.
            print("WARNING: rollout reached move limit")
        return 1 - 2 * (state[1] % 2)

    def get_move(self, state):
        """Runs all playouts sequentially and returns the most visited action.
        Arguments:
        state -- the current state, including both game state and the current player.
        Returns:
        the selected action
        """
        assert(len(state) == GLOBAL_M)
        self._root = TreeNode(None, 1.0)
        for n in range(self._n_playout):
            state_copy = (state.copy(), 0)
            self._playout(state_copy, self._L)

        # chosen action is the *most visited child*, not the highest-value one
        # (they are the same as self._n_playout gets large).
        return max(self._root._children.iteritems(), key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know about the subtree, assuming
        that get_move() has been called already. Siblings of the new root will be garbage-collected.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)
class Cake(object):
	""" n is row limit, m is column limit"""

	def __init__(self, n, m, state=[]):
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

	def add_memory(self, num):
		for i in range(num):
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

	def show(self):
		print(self.board)

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
		return self.get_board()

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
	def human_move(self):
		print(self.board[:self.m])
		action = map(int, raw_input().split())
		while not self.valid(action[0], action[1]):
			print('action is not valid')
			action = map(int, raw_input().split())
		self.decide(action[0], action[1])

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
		if self.board[1] == 1:
			cnt = 0
			for i in range(1, self.m):
				if self.board[i] == 1:
					cnt += 1
			if self.board[0] - 1 < cnt:
				for i in range(self.board[0], self.m):
					self.board[i] = 0;
				return
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

parser = argparse.ArgumentParser()
parser.add_argument('--load', type=str, default=None)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--name', type=str, default=time.strftime("%m_%d_%Y_%H_%M"))
args = parser.parse_args()
BLOCK = 100
# env = gym.make('Pendulum-v0').unwrapped
board = Cake(GLOBAL_N, GLOBAL_M)
board.read_memory('win.in')
board.add_memory(10)
dqn = DeepQNetwork(A_DIM, S_DIM, output_graph=True)
def policy(state):
	global dqn
	action_space = dqn.eval(transform(state))
	valid = option(state)
	dic = {}
	for i in range(len(valid)):
		if valid[i] > 0:
			dic[i] = action_space[i]
	return dic

def rollout_policy(state):
	valid = option(state)
	if len(valid) <= 0:
		return -1
	pos = []
	for i in range(len(valid)):
		if valid[i] > 0:
			pos.append(i)
	if len(pos) > 0:
		return pos[np.random.randint(0, len(pos))]
	else:
		return -1


if args.load is not None:
	print(args.load)
	dqn.restore(args.load)
	print("restore success~")
def encode(x, y):
	return x * GLOBAL_N + y

step = 0
win_cnt = 0
flag = 0
win = []

def test(ti):
	global dqn, GLOBAL_N, board
	print("testing~~~")
	# board = Cake(GLOBAL_N, GLOBAL_M)
	show_ti = 30
	win_time = 0
	for i in range(ti):
		s = transform(board.random_start())
		done = False
		r_sum = 0
		while not done:
			action = dqn.choose_action(s, option(s), GLOBAL_N)
			if i < show_ti:
				print(s[:GLOBAL_M], action/GLOBAL_N, action%GLOBAL_N)
			reward, s_, done = board.move(action/GLOBAL_N, action%GLOBAL_N)
			s = transform(s_)
			r_sum += reward
		if i < show_ti:
			print("end one episode")
		if r_sum > 0:
			win_time += 1
	print("test done~~~")
	print(win_time)

tree = MCTS(policy, rollout_policy)
for episode in range(EP_MAX):
	# initial observation
	s = transform(board.random_start())
	board.show()
	r_sum = 0.0
	while True:

		# RL choose action based on observation
		action = tree.get_move(s[:GLOBAL_M])
		board.decide(action/GLOBAL_N, action%GLOBAL_N)
		board.show()
		if board.over():
			reward, s_, done = board.lose_reward, board.get_board(), True
		else:
			rival_action = tree.get_move(board.get_board())
			board.decide(rival_action/GLOBAL_N, rival_action%GLOBAL_N)
			board.show()
			if board.over():
				reward, s_, done = board.win_reward, board.get_board(), True
			else:
				reward, s_, done = board.step_reward, board.get_board(), False
		r_sum += reward
		step += 1
		s_ = transform(s_)
		dqn.store_transition(s, action, reward, s_)

		if step % BATCH == BATCH - 1:
			print('~~~learning~~~')
			dqn.learn()

		# swap observation
		s = s_

		# break while loop when end of this episode
		if done:
			break
	if r_sum > 0:
		win_cnt += 1
	if episode % BLOCK == 0:
		print(win_cnt)		
			# board.add_memory()
		win.append(win_cnt)
		win_cnt = 0
		# plt.plot(np.arange(len(win)), win)
		# plt.xlabel('Episode');plt.ylabel('win time')
		# plt.savefig("pic/{}.jpg".format(args.name))
	if (episode+1) % BLOCK == 0:
		test(2000)
		path = 'save/{}_{}.ckpt'.format(args.name, episode+1)
		# print(path)
		dqn.save(path)	
