import numpy as np
import tensorflow as tf

from baselines.common.distributions import make_pdtype

def conv_layer(inputs, filters, kernel, strides, gain=1.0):
    return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel, strides=(strides,strides), activation=tf.nn.relu, kernel_initializer=tf.orthogonal_initializer(gain=gain))
    
def fc_layer(inputs, nodes, activation_fn=tf.nn.relu, gain=1.0):
    return tf.layers.dense(inputs=inputs, units=nodes, activation=activation_fn, kernel_initializer=tf.orthogonal_initializer(gain))
    
### Neural Network Class
# init: constructs a convolutional network with actor and critic streams
# step: given a state, return recommended action and value of that state
# value: given a state, return value of that state
# select_action: given a state, return action of that state

class A2CNetwork(object):
    def __init__(self, sess, observation_space, action_space, reuse = False):
        gain = np.sqrt(2)
        
        self.pdtype = make_pdtype(action_space)
        
        height, width, channel = observation_space.shape
        observation_shape = (height, width, channel)

        inputs = tf.placeholder(tf.float32, [None, *observation_shape], name="input")
        
        scaled_images = tf.cast(inputs, tf.float32) / 255.
        
        with tf.variable_scope("model", reuse = reuse):
            conv1 = conv_layer(scaled_images, 32, 8, 4, gain)
            conv2 = conv_layer(conv1, 64, 4, 2, gain)
            conv3 = conv_layer(conv2, 64, 3, 1, gain)
            conv_flatten = tf.layers.flatten(conv3)
            common_fc = fc_layer(conv_flatten, 512, gain=gain)
            
            # self.pi - logits, self.pd - prob distribution
            self.pd, self.pi = self.pdtype.pdfromlatent(common_fc, init_scale=0.01)
            
            # value function
            vf = fc_layer(common_fc, 1, activation_fn=None)[:, 0]
            
        self.initial_state = None
        
        a0 = self.pd.sample()
        
        
        def step(state_in):
            action, value = sess.run([a0, vf], feed_dict={inputs: state_in})
            return action,value
        
        def value(state_in):
            value = sess.run(vf, {inputs: state_in})
            return value
        
        def select_action(state_in):
            action = sess.run(a0, {inputs: state_in})
            return action
        
        self.inputs = inputs
        self.vf = vf
        self.step = step
        self.value = value
        self.select_action = select_action
        