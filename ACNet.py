import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()

#parameters for training
GRAD_CLIP              = 1000.0
KEEP_PROB1             = 1 # was 0.5
KEEP_PROB2             = 1 # was 0.7
RNN_SIZE               = 512
GOAL_REPR_SIZE         = 12

#Used to initialize weights for policy and value output layers (Do we need to use that? Maybe not now)
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

class ACNet:
    def __init__(self, scope, a_size, trainer,TRAINING,GRID_SIZE,GLOBAL_NET_SCOPE):
        with tf.compat.v1.variable_scope(str(scope)+'/qvalues'):
            #The input size may require more work to fit the interface.
            self.inputs = tf.compat.v1.placeholder(shape=[None,4,GRID_SIZE,GRID_SIZE], dtype=tf.float32)
            self.goal_pos=tf.compat.v1.placeholder(shape=[None,3],dtype=tf.float32)
            self.myinput = tf.transpose(self.inputs, perm=[0,2,3,1])
            self.policy, self.value, self.state_out, self.state_in, self.state_init, self.blocking, self.on_goal,self.valids = self._build_net(self.myinput,self.goal_pos,RNN_SIZE,TRAINING,a_size)
        if TRAINING:
            self.actions                = tf.compat.v1.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot         = tf.one_hot(self.actions, a_size, dtype=tf.float32)
            self.train_valid            = tf.compat.v1.placeholder(shape=[None,a_size], dtype=tf.float32)
            self.target_v               = tf.compat.v1.placeholder(tf.float32, [None], 'Vtarget')
            self.advantages             = tf.compat.v1.placeholder(shape=[None], dtype=tf.float32)
            self.target_blockings       = tf.compat.v1.placeholder(tf.float32, [None])
            self.target_on_goals        = tf.compat.v1.placeholder(tf.float32, [None])
            self.responsible_outputs    = tf.reduce_sum(self.policy * self.actions_onehot, [1])
            self.train_value            = tf.compat.v1.placeholder(tf.float32, [None])
            self.optimal_actions        = tf.compat.v1.placeholder(tf.int32,[None])
            self.optimal_actions_onehot = tf.one_hot(self.optimal_actions, a_size, dtype=tf.float32)

            
            # Loss Functions
            self.value_loss    = tf.reduce_sum(self.train_value*tf.square(self.target_v - tf.reshape(self.value, shape=[-1])))
            self.entropy       = - tf.reduce_sum(self.policy * tf.math.log(tf.clip_by_value(self.policy,1e-10,1.0)))
            self.policy_loss   = - tf.reduce_sum(tf.math.log(tf.clip_by_value(self.responsible_outputs,1e-15,1.0)) * self.advantages)
            self.valid_loss    = - tf.reduce_sum(tf.math.log(tf.clip_by_value(self.valids,1e-10,1.0)) *\
                                self.train_valid+tf.math.log(tf.clip_by_value(1-self.valids,1e-10,1.0)) * (1-self.train_valid))
            self.blocking_loss = - tf.reduce_sum(self.target_blockings*tf.math.log(tf.clip_by_value(self.blocking,1e-10,1.0))\
                                      +(1-self.target_blockings)*tf.math.log(tf.clip_by_value(1-self.blocking,1e-10,1.0)))
            self.on_goal_loss = - tf.reduce_sum(self.target_on_goals*tf.math.log(tf.clip_by_value(self.on_goal,1e-10,1.0))\
                                      +(1-self.target_on_goals)*tf.math.log(tf.clip_by_value(1-self.on_goal,1e-10,1.0)))
            self.loss          = 0.5 * self.value_loss + self.policy_loss + 0.5*self.valid_loss \
                            - self.entropy * 0.01 +.5*self.blocking_loss
            self.imitation_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(self.optimal_actions_onehot,self.policy))
            
            # Get gradients from local network using local losses and
            # normalize the gradients using clipping
            local_vars         = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope+'/qvalues')
            self.gradients     = tf.compat.v1.gradients(self.loss, local_vars)
            self.var_norms     = tf.linalg.global_norm(local_vars)
            grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, GRAD_CLIP)

            # Apply local gradients to global network
            global_vars        = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, GLOBAL_NET_SCOPE+'/qvalues')
            self.apply_grads   = trainer.apply_gradients(zip(grads, global_vars))

            #now the gradients for imitation loss
            self.i_gradients     = tf.compat.v1.gradients(self.imitation_loss, local_vars)
            self.i_var_norms     = tf.linalg.global_norm(local_vars)
            i_grads, self.i_grad_norms = tf.clip_by_global_norm(self.i_gradients, GRAD_CLIP)

            # Apply local gradients to global network
            self.apply_imitation_grads   = trainer.apply_gradients(zip(i_grads, global_vars))
        print("Hello World... From  "+str(scope))     # :)

    def _build_net(self,inputs,goal_pos,RNN_SIZE,TRAINING,a_size):
        w_init   = tf.compat.v1.keras.initializers.VarianceScaling()
        conv_kwargs = dict(kernel_initializer=w_init, padding="same", data_format="channels_last")

        conv1    =  tf.keras.layers.Conv2D(filters=RNN_SIZE//4, kernel_size=(3, 3), strides=1, activation=tf.nn.relu, **conv_kwargs)(inputs)
        conv1a   =  tf.keras.layers.Conv2D(filters=RNN_SIZE//4, kernel_size=(3, 3), strides=1, activation=tf.nn.relu, **conv_kwargs)(conv1)
        conv1b   =  tf.keras.layers.Conv2D(filters=RNN_SIZE//4, kernel_size=(3, 3), strides=1, activation=tf.nn.relu, **conv_kwargs)(conv1a)
        pool1    =  tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, data_format="channels_last")(conv1b)
        conv2    =  tf.keras.layers.Conv2D(filters=RNN_SIZE//2, kernel_size=(3, 3), strides=1, activation=tf.nn.relu, **conv_kwargs)(pool1)
        conv2a   =  tf.keras.layers.Conv2D(filters=RNN_SIZE//2, kernel_size=(3, 3), strides=1, activation=tf.nn.relu, **conv_kwargs)(conv2)
        conv2b   =  tf.keras.layers.Conv2D(filters=RNN_SIZE//2, kernel_size=(3, 3), strides=1, activation=tf.nn.relu, **conv_kwargs)(conv2a)
        pool2    =  tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, data_format="channels_last")(conv2b)
        conv3    =  tf.keras.layers.Conv2D(filters=RNN_SIZE-GOAL_REPR_SIZE, kernel_size=(2, 2), strides=1, activation=None, padding="valid", data_format="channels_last", kernel_initializer=w_init)(pool2)

        flat     = tf.nn.relu(tf.keras.layers.Flatten()(conv3))
        goal_layer = tf.keras.layers.Dense(GOAL_REPR_SIZE, activation=tf.nn.relu, kernel_initializer=w_init)(goal_pos)
        hidden_input=tf.concat([flat,goal_layer],1)
        h1 = tf.keras.layers.Dense(RNN_SIZE, activation=tf.nn.relu, kernel_initializer=w_init)(hidden_input)
        d1 = tf.keras.layers.Dropout(rate=1-KEEP_PROB1)(h1, training=TRAINING)
        h2 = tf.keras.layers.Dense(RNN_SIZE, activation=None, kernel_initializer=w_init)(d1)
        d2 = tf.keras.layers.Dropout(rate=1-KEEP_PROB2)(h2, training=TRAINING)
        self.h3 = tf.nn.relu(d2+hidden_input)
        #Recurrent network for temporal dependencies
        lstm_cell = tf.keras.layers.LSTMCell(RNN_SIZE)
        from collections.abc import Sequence
        if isinstance(lstm_cell.state_size, Sequence):
            state_sizes = list(lstm_cell.state_size)
        else:
            state_sizes = [lstm_cell.state_size, lstm_cell.state_size]
        if len(state_sizes) == 2:
            c_size, h_size = state_sizes
        else:
            c_size = h_size = state_sizes[0]
        c_init = np.zeros((1, c_size), np.float32)
        h_init = np.zeros((1, h_size), np.float32)
        state_init = [c_init, h_init]
        c_in = tf.compat.v1.placeholder(tf.float32, [None, c_size])
        h_in = tf.compat.v1.placeholder(tf.float32, [None, h_size])
        state_in = (c_in, h_in)
        rnn_in = tf.expand_dims(self.h3, axis=1)
        rnn_layer = tf.keras.layers.RNN(lstm_cell, return_sequences=True, return_state=True)
        lstm_outputs, lstm_c, lstm_h = rnn_layer(rnn_in, initial_state=[c_in, h_in], training=TRAINING)
        state_out = (lstm_c, lstm_h)
        self.rnn_out = tf.reshape(lstm_outputs, [-1, RNN_SIZE])

        policy_layer = tf.keras.layers.Dense(a_size,
                                             activation=None,
                                             use_bias=False,
                                             kernel_initializer=normalized_columns_initializer(1./float(a_size)))(self.rnn_out)
        policy       = tf.nn.softmax(policy_layer)
        policy_sig   = tf.sigmoid(policy_layer)
        value        = tf.keras.layers.Dense(1,
                                             activation=None,
                                             use_bias=False,
                                             kernel_initializer=normalized_columns_initializer(1.0))(self.rnn_out)
        blocking      = tf.keras.layers.Dense(1,
                                              activation=tf.nn.sigmoid,
                                              use_bias=False,
                                              kernel_initializer=normalized_columns_initializer(1.0))(self.rnn_out)
        on_goal      = tf.keras.layers.Dense(1,
                                             activation=tf.nn.sigmoid,
                                             use_bias=False,
                                             kernel_initializer=normalized_columns_initializer(1.0))(self.rnn_out)

        return policy, value, state_out ,state_in, state_init, blocking, on_goal,policy_sig
