# 增强学习求解组合游戏
## 模型

一个 $n \times m$大小的网格，两个人轮流行动，每人每次挑选一个格子，将这个格子及其右上方的所有格子全部删掉，谁选到最左下角的格子谁输。

## dqn.py

此文件用于生成网络。基于 Deep Q Learning 的方法进行训练。网络的输入为棋盘的特征，输出为每一个动作的打分，输入层和输出层中间为两层全连接层。因为棋盘每一列的格子数是单调减少的，我们用每一列还剩多少个格子来表示一个棋盘。其他使用的特征:

* 总共还剩多少个格子
* 还剩多少列
* 每列与其右列格数是否相同
* 最长连续等格列有多长
* 每列与其左列和右列是否构成等差数列
* 每列所剩格数是奇数还是偶数

举例，当棋盘大小为$n \cdot n$时，网络的输入个数为$4n + 2$,输出个数为$n \cdot n$.

一些超参的解释：

* e_greedy, 决定sample动作时以多大的概率选择网络打分最高的动作，否则完全随机sample一个动作
* replace_target_iter, 每更新多少轮用q_eval替换q_target.
* memory_size, DQN的记忆库的大小。
* batch_size, 一次更新sample多少个样本
* epsilon_increment, 训练过程逐渐增加e_greedy的大小以增大exploitability减小exploration.

## run_dqn.py

此文件主要调用dqn.py进行训练。一些函数的解释：

* option: 输入棋盘，输出该局面下所有合法的动作。
* read_memory: 从文件中读入一些必败点。
* decide: 根据输入的动作,更改棋盘。
* valid: 用于当前局面是否合法。
* rival: 在一些简单的模式，输出最优的动作。其他情况，随机生成一个动作。

## mct_dqn.py

加入蒙特卡洛搜索树进行训练。蒙特卡洛搜索树的参数:

* n_playout, 每次生成一个动作执行多少次模拟。
* rollout_limit, 每次模拟最多迭代多深。
* playout_depth, 深度超过多少时采用粗糙的策略
* c_puct, 越高搜索树中节点的打分越接近于预设值，用于控制exploitability和exploration

parser 中的load用于指定载入的模型，name用于指定训练生成的网络的名字，不指定则以文件运行的时间命名，其他文件类似。

## ppo_easy.py

用单线程的Proximal Policy Optimization算法进行训练。这里对PPO的两种实现方式:在loss中加入更新前输出动作的概率分布与更新后输出动作的概率分布的
KL距离，以及对更新后动作概率与更新前动作概率的比例进行截断，均进行了实现。
## a3c.py
用Asynchronous Advantage Actor Critic算法进行训练。A3C算法每个线程对应一个actor和critic,可以并行执行，速度较快。

