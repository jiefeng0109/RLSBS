"""
Asynchronous Advantage Actor Critic (A3C) with discrete action space for hyperspectral band selesction
自适应波段的版本
2020年3月1日修改
"""

#import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import time
import torch
import argparse

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import cohen_kappa_score
from processing_library import load_data, one_hot, disorder, next_batch
from processing_library import contrary_one_hot, expend, windowFeature, pca_trans_expend
from net import Net
from hsi_env import HSIEnv


parser = argparse.ArgumentParser()

parser.add_argument('--data_name', type=str, default='Indian_pines',
                    help='数据集名称')
parser.add_argument('--patch_size', type=int, default=31,
                    help='窗口大小')
parser.add_argument('--num_band_seclection', type=int, default=100 ,
                    help='要选择的波段数目')
parser.add_argument('--N_WORKERS', type=int, default=1,
                    help='线程数目')
parser.add_argument('--MAX_GLOBAL_EP', type=int, default=2000,
                    help='最大迭代代数')
parser.add_argument('--UPDATE_GLOBAL_ITER', type=int, default=200,#200
                    help='UPDATE_GLOBAL_ITER') 
parser.add_argument('--LR_A', type=float, default=0.001,
                    help='Actor的学习率')
parser.add_argument('--LR_C', type=float, default=0.001,
                     help='Critic的学习率')
parser.add_argument('--REWARD_RATE', type=float, default=100,
                    help='奖励倍率')
parser.add_argument('--LAMBDA_COST', type=float, default=-0.,
                    help='波段数目惩罚')
parser.add_argument('--eval_net_path', type=str, default='./checkpoint/Indian_pines.t7',
                    help='evalnet的权值保存位置')
parser.add_argument('--out_put_path', type=str, default='./output/Indian_pines',
                    help='输出位置')
FLAGS, unparsed = parser.parse_known_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
#tf_config = tf.ConfigProto()
#tf_config.gpu_options.per_process_gpu_memory_fraction = 0.2  # 分配50%
###############################################################################
# data_norm,labels_ori,x_train,y_train,train_loc,x_test,y_test,test_loc=load_data('Indian_pines')
data_norm, labels_ori, y_train, train_loc, y_test, test_loc, y_val, val_loc, ulab_loc = load_data(
    FLAGS.data_name)
nrows, ncols, ndim = data_norm.shape

ulab_loc = val_loc
val_num = int(.5* train_loc.shape[0])
y_val = y_train[val_num:]
val_loc = train_loc[val_num:]

dim_input = ndim
# dim_input = ndim
num_classification = int(np.max(labels_ori))  # 类别数
w = FLAGS.patch_size  # 图像块大小
# num_epoch = 30000  # 训练循环次数
num_band_seclection = FLAGS.num_band_seclection  # 要选择的波段数numOut = 64


###############################################################################
# load data
data_norm = expend(data_norm, w)
labels_ori_ex = expend(labels_ori[:, :, np.newaxis], w)
#data2_norm = expend(data2_norm,w)
dim_input = np.shape(data_norm)[2]

Y_train = one_hot(y_train, num_classification)
Y_test = one_hot(y_test, num_classification)
Y_val = one_hot(y_val, num_classification)

train_loc, Y_train = disorder(train_loc, Y_train)
test_loc, Y_test = disorder(test_loc, Y_test)
val_loc, Y_val = disorder(val_loc, Y_val)
np.random.shuffle(ulab_loc)
###############################################################################
# load net
net = Net(input_bands=ndim, num_classification=num_classification)
#criterion = nn.CrossEntropyLoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#device =torch.device("cpu")
print(device)
net.to(device)
print('===> Try resume from checkpoint')
if os.path.isdir('checkpoint'):
    try:
        checkpoint = torch.load(FLAGS.eval_net_path)
        net.load_state_dict(checkpoint['state'])        # 从字典中依次读取
        start_epoch = checkpoint['epoch']
        print('===> Load last checkpoint data')
    except FileNotFoundError:
        print('Can\'t found '+FLAGS.eval_net_path)
else:
    start_epoch = 0
    print('===> Start from scratch')
###############################################################################
# get accuracy with all features


data_val = {"data_norm": data_norm,
            "X_valid_loc": val_loc, #!!
            "Y_valid": Y_val,  
            "w": w,
            "unl_loc":ulab_loc}
# env = HSIEnv(net, data_val, ndim, num_band_seclection, None, lambada_cost=FLAGS.LAMBDA_COST)
#selected_bands = []
#nb_states = ndim  # 状态向量
#nb_actions = ndim + 1 # 动作向量 = 波段数目+1


# GAME = 'CartPole-v0'
OUTPUT_GRAPH = True
LOG_DIR = './log'
#N_WORKERS = multiprocessing.cpu_count()
N_WORKERS = FLAGS.N_WORKERS
MAX_GLOBAL_EP = FLAGS.MAX_GLOBAL_EP
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = FLAGS.UPDATE_GLOBAL_ITER
GAMMA = .999
ENTROPY_BETA = 0.001
LR_A = FLAGS.LR_A   # learning rate for actor
LR_C = FLAGS.LR_C    # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_ACC = []
GLOBAL_EP = 0
# start_time =time.time()
filename = str(int(time.time()))+"_"+str(FLAGS.num_band_seclection)+"_bands"
if not os.path.exists(FLAGS.out_put_path+"/"+filename):
    os.makedirs(FLAGS.out_put_path+"/"+filename)
start = time.time()
start_all = time.time()
# env = gym.make(GAME)
N_S = ndim
N_A = ndim # 动作向量 = 波段数目+1


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
                self.v_target = tf.placeholder(
                    tf.float32, [None, 1], 'Vtarget')

                self.a_prob, self.v, self.a_params, self.c_params = self._build_net(
                    scope)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(tf.log(self.a_prob + 1e-5) * tf.one_hot(
                        self.a_his, N_A, dtype=tf.float32), axis=1, keep_dims=True)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5),
                                             axis=1, keep_dims=True)  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(
                        self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(
                        self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(
                        zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(
                        zip(self.c_grads, globalAC.c_params))
        self.saver = tf.train.Saver()

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 64, tf.nn.relu6,
                                  kernel_initializer=w_init, name='la')
#            l_a_norm = tf.contrib.layers.batch_norm(l_a, 0.9, epsilon=1e-5, activation_fn = None, is_training=True)
            l_a_2 = tf.layers.dense(
                l_a, 64, tf.nn.relu6, kernel_initializer=w_init, name='la_2')
#            l_a_2_norm = tf.contrib.layers.batch_norm(l_a_2, 0.9, epsilon=1e-5, activation_fn = None, is_training=True)
            a_prob = tf.layers.dense(
                l_a_2, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 64, tf.nn.relu6,
                                  kernel_initializer=w_init, name='lc')
            l_c_2 = tf.layers.dense(
                l_c, 32, tf.nn.relu6, kernel_initializer=w_init, name='lc_2')
            v = tf.layers.dense(
                l_c_2, 1, kernel_initializer=w_init, name='v')  # state value
        a_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return a_prob, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        # local grads applies to global net
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)

    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        prob_weights = SESS.run(self.a_prob, feed_dict={
                                self.s: s[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def save_model(self, save_path):
        self.saver.save(SESS, save_path)

    def restore_model(self, save_path):
        self.saver.restore(SESS, save_path)
#    def choose_action_test(self,s):#在测试阶段，确定性的选择动作
#        prob_weights = SESS.run(self.a_prob, feed_dict={self.s: s[np.newaxis, :]})
#        action = np.argmax(prob_weights.ravel())
#        return action


class Worker(object):
    def __init__(self, name, globalAC):
        self.env = HSIEnv(net, data_val, ndim, num_band_seclection, num_classification,reward_rare=FLAGS.REWARD_RATE, lambada_cost=FLAGS.LAMBDA_COST)
        self.name = name
        self.AC = ACNet(name, globalAC)
        self.max_r_bands = None

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP, GLOBAL_ACC,start
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        r_max, acc_max = 0, 0

        # training prase
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            while True:
                # if self.name == 'W_0':
                #     self.env.render()
                a = self.AC.choose_action(s)
#                print(a)
                s_, r, done, info = self.env.step(a,GLOBAL_EP)
#                print(r)
#                if done: r = -5
                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = r   # terminal
                    else:
                        v_s_ = SESS.run(
                            self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(
                        buffer_s), np.array(buffer_a), np.vstack(buffer_v_target)
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
                        GLOBAL_ACC.append(info)
                    else:
                        GLOBAL_ACC.append(info)
                        # GLOBAL_ACC.append(0.99*GLOBAL_ACC[-1]+0.01 * info)
                        GLOBAL_RUNNING_R.append(
                            0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * ep_r)
                        if GLOBAL_EP % 50 == 0:
                            time_t = time.time()-start
                            start = time.time()
                            print("ep ", GLOBAL_EP, "/", MAX_GLOBAL_EP)
                            print(
                                self.name,
                                "Ep:", GLOBAL_EP,
                                "| Ep_r: %.2f" % GLOBAL_RUNNING_R[-1],
                                "| loss:%.3f"%(info),
                                "| %d bands"%(np.sum(s)),
                                "| %.1f"%(time_t)+"s",
                            )

                        if r > r_max:
                            r_max = r
                            acc_max = info
                            self.max_r_bands = s

                    GLOBAL_EP += 1
                    break
        if  self.name == "W_0":
            # self.AC.save_model(FLAGS.out_put_path+"/" +
            #                 filename+"/model/weights.ckpt")
            # testing prase
            s = self.env.reset()
            buffer_s_t, buffer_a_t, buffer_r_t = [], [], []
            buffer_s_t.append(s)
            with open(FLAGS.out_put_path+"/"+filename+"/result.txt", "w") as f:
                while True:
                    a = self.AC.choose_action(s)
                    s_, r, done, info = self.env.step(a,3000)
                    buffer_s_t.append(s_)
                    buffer_a_t.append(a)
                    buffer_r_t.append(r)
                    s = s_
                    f.write("s: "+str([i for i in range(len(list(s))) if s[i]
                                    == 1])+"\nr: "+str(r)+" info: "+str(info)+"\n")
                    if done:
                        np.save(FLAGS.out_put_path+"/ac3_bands", s_)
                        # np.save("ac3_bands", s_)
                        print("r_max,", info)
                        print("Done")
                        break


if __name__ == "__main__":
    if not os.path.exists(FLAGS.out_put_path+"/"+filename+"/model"):
        os.makedirs(FLAGS.out_put_path+"/"+filename+"/model")
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

#    if OUTPUT_GRAPH:
#        if os.path.exists(LOG_DIR):
#            shutil.rmtree(LOG_DIR)
#        tf.summary.FileWriter(LOG_DIR, SESS.graph)

    worker_threads = []
    for worker in workers:
        def job(): return worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)

    end = time.time()
    with open(FLAGS.out_put_path+"/"+filename+"/conf.txt", "w") as f:
        f.write(str(FLAGS)+"\n")
        f.write("training time: "+str(int(end-start_all)))

    print("time :%d" % (end-start_all))
    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')

    plt.savefig(FLAGS.out_put_path+"/"+filename+"/reward.jpg")
    plt.close()
    # print(GLOBAL_ACC)
    plt.plot(
        np.arange(len(np.reshape(GLOBAL_ACC, [-1, ]))), np.reshape(GLOBAL_ACC, [-1, ]))
    plt.xlabel('step')
    plt.ylabel('loss')
    #    plt.show()
    plt.savefig(FLAGS.out_put_path+"/"+filename+"/loss.jpg")
    #    exit()