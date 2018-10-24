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
        Concat action input and state input, predict difference
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

    def A_star_planner(self, cur_state, num_samples):
        '''
        A star planner: Obtain a sequence of actions and use the given heuristics to choose the best action
        :param cur_state: current observation
        :param num_samples: number discrete observation you would like to have
        :return:
        '''

        adv_list = np.zeros([num_samples])
        old_obs = np.asarray([cur_state for i in range(num_samples)])
        new_obs = old_obs
        # Obstain sequence length of 10 of action and states
        for i in range(10):
            action = np.asarray([-2 + 4/num_samples * i for i in range(num_samples)])[:,np.newaxis]#(np.random.rand(num_samples, self.action_size)-0.5)*4
            if i == 0:
                action_list = action
            diff = self.sess.run(self.output, feed_dict={self.input_s: np.asarray(new_obs).reshape([-1,self.state_size]),
                                                            self.input_a: np.asarray(action).reshape([-1,self.action_size])})
            new_obs = old_obs + diff
            angle = np.arccos(new_obs[:,0])
            heuristics = -((((angle+np.pi) % (2*np.pi)) - np.pi) **2 + new_obs[:,2]**2*0.1 + 0.001* action[:,0]**2)
            # cummulate the heuristics for each sequence of actions and states
            adv_list[:] += heuristics
        # choose the action by the maximum heuristics
        index = np.argmax(adv_list)
        return action_list[index]

    def build_optimizer(self):
        '''
         MSE loss, predict the difference between current state and next state
         AdamOptimizer with 0.001 learning rate
        :return:
        '''

        self.difference = tf.reduce_mean(self.output, self.target_s)
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
        vp_t_s = np.asarray(self.buffer_t_s) - vp_s

        _, difference, summary = self.sess.run([self.train_op, self.difference, self.merged],
            feed_dict = {self.input_a: vp_a, self.input_s:vp_s, self.target_s: vp_t_s})
        self.writter.add_summary(summary)
        return difference


if __name__ == '__main__':
    '''
    Using a method, predict the difference of current state and next state
    '''
    ops = parseOptions()

    env = gym.make('Pendulum-v0').unwrapped
    sess = tf.Session()

    Agent = agent(env.observation_space.shape[0], env.action_space.shape[0],sess)
    diff_s = []
    if ops.load_model:
        Agent.saver.restore(sess,tf.train.latest_checkpoint(checkpoint_dir="./model/"))

    for episode in range(1000):
        state = env.reset()
        ep_r = 0
        Agent.buffer_r, Agent.buffer_s, Agent.buffer_a, Agent.buffer_t_s = [], [], [], []
        for step in range(200):
            env.render()

            action = Agent.A_star_planner(state,num_samples=1000)
            next_state, reward, done, _ = env.step(action)
            Agent.store_transition(state, action, reward,next_state)
            state = next_state

            ep_r += reward
            if (step + 1)%32 == 0 or step == 199:
                difference = Agent.update()
        if ops.save_model:
            Agent.saver.save(sess, "./model/model.ckpt")

        diff_s.append(difference)
        print("episode_reward : {}, training_state_error: {}".format(ep_r,difference))
    plt.plot(diff_s)
    plt.show()