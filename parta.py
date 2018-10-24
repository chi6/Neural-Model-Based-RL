'''
Implemented by Chi Zhang.
Please take a look at my github repo,
I have also done some project about model-based meta reinforcement learning
(Using maml to learn a transition model for RL).
'''

import gym
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import optparse

LR = 0.001

def parseOptions():
    optParser = optparse.OptionParser()

    optParser.add_option('-s', '--save',action='store',
                         type='string',dest='save_model',default="True",
                         metavar="R", help='Reward for living for a time step (default %default)')
    optParser.add_option('-l', '--load', action='store',
                         type='string', dest='load_model',default="True",
                         help='Load model from checkpoint')

    opts, args = optParser.parse_args()

    return opts

class agent(object):
    def __init__(self, state_size, action_size, sess):
        self.state_size = state_size
        self.action_size = action_size
        self.sess = sess

        self.build_net()
        self.build_optimizer()
        self.buffer_s, self.buffer_a, self.buffer_r, self.buffer_t_s = [], [], [], []

        self.writter = tf.summary.FileWriter('./model/',self.sess.graph)
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def build_net(self):
        '''
        Build input and a two layers network
        Concat action input and state input, predict next state
        :return:
        '''
        self.input_a = tf.placeholder(tf.float32, shape= [None, self.action_size], name= 'action')
        self.input_s = tf.placeholder(tf.float32, shape= [None,self.state_size], name = 'state')
        self.target_s = tf.placeholder(tf.float32, shape= [None,self.state_size], name = 'target_state')

        concat_input = tf.concat((self.input_a, self.input_s), axis= 1)
        with tf.variable_scope('model_pred'):
            layer1 = tf.layers.dense(concat_input, 128, activation=tf.nn.relu)
            layer2 = tf.layers.dense(layer1, 128, activation=tf.nn.relu)
            self.output = tf.layers.dense(layer2, self.state_size, activation= None)

    def build_optimizer(self):
        '''
         MSE loss, predict next state
         AdamOptimizer with 0.001 learning rate
        :return:
        '''

        self.difference = tf.reduce_mean(tf.abs(tf.subtract(self.output, self.target_s)))
        loss = tf.reduce_mean(tf.square(tf.subtract(self.output, self.target_s)))
        self.train_op = tf.train.AdamOptimizer(learning_rate= LR).minimize(loss)

        tf.summary.scalar("loss", loss)
        tf.summary.scalar("training_error", self.difference)
        self.merged = tf.summary.merge_all()

    def store_transition(self, s, a, r,t_s):
        '''
        Store transition data
        :param s: current state
        :param a: current action
        :param r: reward from next state
        :param t_s: next state
        :return:
        '''
        self.buffer_s.append(s)
        self.buffer_a.append(a)
        self.buffer_r.append(r)
        self.buffer_t_s.append(t_s)


    def update(self):
        '''
        Feed rollout buffer to network and do optimization
        Obtain current training error
        :return:
        '''
        vp_s = np.asarray(self.buffer_s)
        vp_a = np.asarray(self.buffer_a)
        vp_t_s = np.asarray(self.buffer_t_s)

        _, difference, summary = self.sess.run([self.train_op, self.difference, self.merged],
            feed_dict = {self.input_a: vp_a, self.input_s:vp_s, self.target_s: vp_t_s})
        self.writter.add_summary(summary)
        return difference


if __name__ == '__main__':
    ops = parseOptions()


    env = gym.make('Pendulum-v0').unwrapped
    sess = tf.Session()

    Agent = agent(env.observation_space.shape[0], env.action_space.shape[0],sess)
    if ops.load_model:
        Agent.saver.restore(sess,tf.train.latest_checkpoint(checkpoint_dir="./model/"))


    diff_s = []
    for episode in range(8000):
        state = env.reset()
        Agent.buffer_r,Agent.buffer_s,Agent.buffer_a, Agent.buffer_t_s = [], [],[],[]
        for step in range(200):
            action = np.random.uniform(-2,2)
            next_state, reward, done, _ = env.step([action])
            Agent.store_transition(state, action, reward, next_state)

            if (step + 1)%32 == 0 or step == 199:
                difference = Agent.update()
                if ops.save_model:
                    Agent.saver.save(sess,"./model/model.ckpt")
        diff_s.append(difference)
        print(difference)
    plt.plot(diff_s)
    plt.show()