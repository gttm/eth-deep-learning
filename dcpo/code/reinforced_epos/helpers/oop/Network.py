import numpy as np
import tensorflow as tf
import sonnet as snt
import keras


def weight_variable(name, shape):
    W = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    return W


def bias_variable(name, shape):
    b = tf.get_variable(name, initializer=tf.constant(0.1, shape=shape))
    return b


class Network():
    def __init__(self, dataset, total_actions, scope: str, trainer, lstm_size=40):
        shape = np.shape(dataset)
        self.users = shape[0]
        self.plans = shape[1]
        self.scope = scope
        self.timesteps = shape[2]
        self.total_actions = total_actions
        with tf.variable_scope(self.scope):
            # graph input which is batch x users x state

            self.input = tf.placeholder(shape=[None, self.users, self.timesteps], dtype=tf.float32)

            # Convolution on timesteps
            conv_1_filter = [1, 12]
            conv_1_stride = [1, 12]

            self.add_channels = tf.expand_dims(self.input, axis=3)  # add an empty channel for layer to work
            self.conv_1 = keras.layers.Conv2D(80, (1, 4), strides=(1, 3), padding='valid',
                                              dilation_rate=1, activation="relu", use_bias=True,
                                              kernel_initializer='glorot_uniform', bias_initializer='zeros',
                                              kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                                              kernel_constraint=None, bias_constraint=None)(self.add_channels)
            self.conv_2 = keras.layers.Conv2D(40, (1, 3), strides=(1, 2), padding='valid', dilation_rate=1,
                                              activation="relu", use_bias=True, kernel_initializer='glorot_uniform',
                                              bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                                              activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(
                self.conv_1)
            self.conv_3 = keras.layers.Conv2D(20, (1, 3), strides=(1, 2), padding='valid',
                                              dilation_rate=1, activation="relu",
                                              use_bias=True, kernel_initializer='glorot_uniform',
                                              bias_initializer='zeros',
                                              kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                                              kernel_constraint=None, bias_constraint=None)(self.conv_2)

            # tf.squeeze(self.conv_3, axis=1)
            self.squeeze_channels = tf.reshape(self.conv_3, [-1, self.users, 11 * 20])

            # keras.layers.CuDNNLSTM(units, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
            #  bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, return_state=False, stateful=False)

            # just one state all repeat the states of the whole batch?

            self.lstm_h_prev = tf.placeholder(shape=[1, lstm_size], dtype=tf.float32)
            self.lstm_c_prev = tf.placeholder(shape=[1, lstm_size], dtype=tf.float32)

            self.lstm_h_init = np.zeros([1, lstm_size])
            self.lstm_c_init = np.zeros([1, lstm_size])
            self.state_init = [self.lstm_h_init, self.lstm_c_init]

            self.state_in = np.array([self.lstm_h_prev, self.lstm_c_prev])

            self.lstm, self.lstm_h, self.lstm_c = keras.layers.LSTM(lstm_size, activation='tanh',
                                                                    recurrent_activation='hard_sigmoid',
                                                                    use_bias=True, kernel_initializer='glorot_uniform',
                                                                    recurrent_initializer='orthogonal',
                                                                    bias_initializer='zeros',
                                                                    unroll=True, return_sequences=True,
                                                                    return_state=True,
                                                                    unit_forget_bias=True, kernel_regularizer=None,
                                                                    recurrent_regularizer=None,
                                                                    implementation=1)(
                self.squeeze_channels, initial_state=[self.state_in[0], self.state_in[1]])

            self.state_out = (self.lstm_h, self.lstm_c)

            # batch x agents x actions


            self.cudnn_lstm = keras.layers.CuDNNLSTM(lstm_size,
                                   kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                                   bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None,
                                   recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                                   kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
                                   return_sequences=True, return_state=False, stateful=False)

            # A second LSTM across network can be put here or LSTM conv2D to capture temporal dependencies
            # Here we should be at batch x agent x actions

            # time dense is indepented application of time
            # self.time_dense = keras.layers.TimeDistributed(keras.layers.Dense(self.actions))(self.lstm_cell)

            self.bid = keras.layers.Bidirectional(self.cudnn_lstm, merge_mode='mul', weights=None)(self.lstm)


            # self.pre_policy = keras.layers.Dense(self.users*self.total_actions)(self.lstm)
            self.pre_policy = keras.layers.Dense(total_actions)(self.bid)
            # self.pre_policy = tf.reshape(self.last_dense, [-1, self.users, self.total_actions])


            # self.lstm_state = keras.layers.LSTM(lstm_size)(self.pre_policy, initial_state=[tf.zeros([1,512]), tf.zeros([1,512])]) #temporal embedding

            # TODO softmax taks into account batch dim as well? check code
            self.policy = tf.nn.softmax(self.pre_policy, dim=2)  # 2 or 1
            # self.weights_value = weight_variable("weights_value", shape=[self.users, self.total_actions])
            self.weights_value = weight_variable("weights_value", shape=[self.users, lstm_size])

            self.value = tf.reduce_sum(tf.multiply(self.bid, self.weights_value))

            # Loss optimizers for the worker networks
            if self.scope != "global":
                self.actions = tf.placeholder(shape=[None, self.users], dtype=tf.int32,
                                              name="actions")  # batch x agents x 1
                self.actions_one_hot = tf.one_hot(self.actions, self.total_actions, dtype=tf.float32,
                                                  axis=2)  # agents x actions
                self.target_v = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="target_value")
                self.advantages = tf.placeholder(
                    shape=[None, 1], dtype=tf.float32,
                    name="advantages")  # this is generated from the discounted reward #batch x 1
                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_one_hot)  # batch x 1

                # Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
                # TODO check if makes sense to take mean entropy
                self.entropy = - tf.reduce_mean(
                    tf.reduce_sum(self.policy * tf.log(self.policy), axis=1))  # mean entropy over agents x 1
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs) * self.advantages)

                self.loss = 10* self.value_loss + self.policy_loss - self.entropy * 10

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

                # Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))


def main():
    dataset = np.random.random([5, 6, 10])
    shape = np.shape(dataset)
    batch_input = [
        [dataset[i, 1, :] for i in range(shape[0])],
        [dataset[i, 2, :] for i in range(shape[0])],
        [dataset[i, 3, :] for i in range(shape[0])],
        [dataset[i, 4, :] for i in range(shape[0])]
    ]
    batch_input = np.array(batch_input)

    net = Network(dataset, 3, "global", None)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = {net.input: batch_input,
                     net.lstm_h_prev: net.lstm_h_init,
                     net.lstm_c_prev: net.lstm_c_init
                     }
        # lstm, lstm_h, lstm_c = sess.run([net.lstm, net.lstm_h, net.lstm_c], feed_dict=feed_dict)
        lstm, last_dense = sess.run([net.lstm, net.pre_policy], feed_dict=feed_dict)

        print(np.shape(lstm))
        print(np.shape(last_dense))

        # print(np.shape(lstm_h))
        # print(np.shape(lstm_c))


        v, p = sess.run([net.value, net.policy], feed_dict=feed_dict)
        print(np.shape(p))
        print(v)


if __name__ == '__main__':
    main()
