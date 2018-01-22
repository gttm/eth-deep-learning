import os
import argparse
import numpy as np
import tensorflow as tf
import reinforced_epos.helpers.dataset as ds
import reinforced_epos.helpers.config as cf

CONV1D_WINDOW = 7
FLAGS = None

def print_and_log(message, log_file):
    print(message)
    log_file.write(message + "\n")
    log_file.flush()

def main(argv):
    # Load the dataset
    dataset_input = ds.get_dataset()
    shape = np.shape(dataset_input)
    NUM_AGENTS = shape[0]
    NUM_PLANS = shape[1]
    NUM_TIMESTEPS = shape[2]
    NUM_ACTIONS = NUM_PLANS
    dataset = tf.constant(dataset_input, dtype=tf.float32, name="dataset")
    # Placehoolder for the agents' states
    states = tf.placeholder(tf.int32, shape=[NUM_AGENTS], name="states")
    
    def weight_variable(name, shape):
        W = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        return W
    
    def bias_variable(name, shape):
        b = tf.get_variable(name, initializer=tf.constant(0.1, shape=shape))
        return b
    
    def conv1d_layer(x, W, b, name=None):
        conv = tf.nn.bias_add(tf.nn.conv1d(x, W, stride=1, padding="SAME"), b, name=name)
        relu = tf.nn.relu(conv)
        return relu
    
    def conv2d_layer(x, W, b, name=None):
        conv = tf.nn.bias_add(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME"), b, name=name)
        relu = tf.nn.relu(conv)
        return relu
    
    def pool_layer(x, name=None):
        pool = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name=name)
        return pool
    
    def fc_layer(x, W, b, name=None):
        fc = tf.nn.bias_add(tf.matmul(x, W), b, name=name)
        relu = tf.nn.relu(fc)
        return relu
    
    # For each agent and for each plan reduce the NUM_TIMESTEPS into a single value
    W_emb = weight_variable(name="W_emb", shape=[NUM_AGENTS, NUM_PLANS, NUM_TIMESTEPS])
    b_emb = bias_variable("b_emb", [NUM_AGENTS, NUM_PLANS])
    
    emb = tf.add(tf.reduce_sum(tf.multiply(dataset, W_emb), axis=2), b_emb, name="emb")
    
    # Filter the embeddings for the current states
    states_emb = tf.reduce_sum(tf.multiply(emb, tf.one_hot(states, NUM_PLANS)), axis=1, name="states_emb")
    
    # Deep Q-Network
    # Architecture: BiLSTM->FC
    if FLAGS.arch == "lstm":
        # Reshape and unstack
        rnn_input = tf.unstack(tf.reshape(states_emb, [NUM_AGENTS, 1, 1], name="rnn_input"))
        
        W_fc = weight_variable("W_fc", [NUM_AGENTS*256*2, NUM_AGENTS*NUM_ACTIONS])
        b_fc = bias_variable("b_fc", [NUM_AGENTS*NUM_ACTIONS])
        cell_fw = tf.contrib.rnn.LSTMCell(num_units=256, initializer=tf.contrib.layers.xavier_initializer())
        cell_bw = tf.contrib.rnn.LSTMCell(num_units=256, initializer=tf.contrib.layers.xavier_initializer())
        
        rnn_output, _, _ = tf.nn.static_bidirectional_rnn(cell_fw, cell_bw, rnn_input, dtype=np.float32)
        rnn_flat = tf.reshape(rnn_output, [1, NUM_AGENTS*256*2])
        fc = tf.nn.bias_add(tf.matmul(rnn_flat, W_fc), b_fc, name="fc")
    
    # Architecture: [CONV]*8->FC
    elif FLAGS.arch == "cnn":
        # Reshape to NHWC format: [batch, in_width, in_channels]
        cnn_input = tf.reshape(states_emb, [1, NUM_AGENTS, 1], name="cnn_input")
        
        W_1_1 = weight_variable("W_1_1", [CONV1D_WINDOW, 1, 32])
        W_1_2 = weight_variable("W_1_2", [CONV1D_WINDOW, 32, 32])
        W_2_1 = weight_variable("W_2_1", [CONV1D_WINDOW, 32, 64])
        W_2_2 = weight_variable("W_2_2", [CONV1D_WINDOW, 64, 64])
        W_3_1 = weight_variable("W_3_1", [CONV1D_WINDOW, 64, 128])
        W_3_2 = weight_variable("W_3_2", [CONV1D_WINDOW, 128, 128])
        W_4_1 = weight_variable("W_4_1", [CONV1D_WINDOW, 128, 256])
        W_4_2 = weight_variable("W_4_2", [CONV1D_WINDOW, 256, 256])
        W_fc = weight_variable("W_fc", [NUM_AGENTS*256, NUM_AGENTS*NUM_ACTIONS])
        b_1_1 = bias_variable("b_1_1", [32])
        b_1_2 = bias_variable("b_1_2", [32])
        b_2_1 = bias_variable("b_2_1", [64])
        b_2_2 = bias_variable("b_2_2", [64])
        b_3_1 = bias_variable("b_3_1", [128])
        b_3_2 = bias_variable("b_3_2", [128])
        b_4_1 = bias_variable("b_4_1", [256])
        b_4_2 = bias_variable("b_4_2", [256])
        b_fc = bias_variable("b_fc", [NUM_AGENTS*NUM_ACTIONS])
        
        conv_1_1 = conv1d_layer(cnn_input, W_1_1, b_1_1, name="conv_1_1")
        conv_1_2 = conv1d_layer(conv_1_1, W_1_2, b_1_2, name="conv_1_2")
        
        conv_2_1 = conv1d_layer(conv_1_2, W_2_1, b_2_1, name="conv_2_1")
        conv_2_2 = conv1d_layer(conv_2_1, W_2_2, b_2_2, name="conv_2_2")
        
        conv_3_1 = conv1d_layer(conv_2_2, W_3_1, b_3_1, name="conv_3_1")
        conv_3_2 = conv1d_layer(conv_3_1, W_3_2, b_3_2, name="conv_3_2")
        
        conv_4_1 = conv1d_layer(conv_3_2, W_4_1, b_4_1, name="conv_4_1")
        conv_4_2 = conv1d_layer(conv_4_1, W_4_2, b_4_2, name="conv_4_2")
        
        conv_flat = tf.reshape(conv_4_2, [1, NUM_AGENTS*256])
        fc = tf.nn.bias_add(tf.matmul(conv_flat, W_fc), b_fc, name="fc")
    
    # Architecture: [FC]*4
    elif FLAGS.arch == "fc":
        # Reshape
        fc_input = tf.reshape(states_emb, [1, NUM_AGENTS], name="fc_input")
        
        W_fc_1 = weight_variable("W_fc_1", [NUM_AGENTS, NUM_AGENTS*32])
        W_fc_2 = weight_variable("W_fc_2", [NUM_AGENTS*32, NUM_AGENTS*64])
        W_fc_3 = weight_variable("W_fc_3", [NUM_AGENTS*64, NUM_AGENTS*128])
        W_fc = weight_variable("W_fc", [NUM_AGENTS*128, NUM_AGENTS*NUM_ACTIONS])
        b_fc_1 = bias_variable("b_fc_1", [NUM_AGENTS*32])
        b_fc_2 = bias_variable("b_fc_2", [NUM_AGENTS*64])
        b_fc_3 = bias_variable("b_fc_3", [NUM_AGENTS*128])
        b_fc = bias_variable("b_fc", [NUM_AGENTS*NUM_ACTIONS])
        
        fc_1 = tf.nn.bias_add(tf.matmul(fc_input, W_fc_1), b_fc_1, name="fc_1")
        fc_2 = tf.nn.bias_add(tf.matmul(fc_1, W_fc_2), b_fc_2, name="fc_2")
        fc_3 = tf.nn.bias_add(tf.matmul(fc_2, W_fc_3), b_fc_3, name="fc_3")
        fc = tf.nn.bias_add(tf.matmul(fc_3, W_fc), b_fc, name="fc")

    Q = tf.reshape(fc, [NUM_AGENTS, NUM_ACTIONS], name="Q")

    # Compute the actions
    actions_indices = tf.argmax(Q, axis=1, output_type=tf.int32, name="actions_indices")
    
    # Loss: Sum((Q_target - Q)^2)
    Q_target = tf.placeholder(tf.float32, shape=[NUM_AGENTS, NUM_ACTIONS], name="Q_target")
    loss = tf.reduce_sum(tf.square(Q_target - Q), name="loss")
    
    # Use Adam for the optimization
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)
    
    # Accumulate summaries
    loss_summary = tf.summary.scalar("loss_summary", loss)
    reward_input = tf.placeholder(tf.float32, name="reward_input")
    reward_summary = tf.summary.scalar("reward_summary", reward_input)
    variance_input = tf.placeholder(tf.float32, name="variance_input")
    variance_summary = tf.summary.scalar("variance_summary", variance_input)
    summary_merged = tf.summary.merge([loss_summary, reward_summary, variance_summary])
    
    # Add ops to save and restore all the variables
    saver = tf.train.Saver()
    
    # Create a local session to run this computation
    with tf.Session() as sess:
        os.makedirs(FLAGS.model_dir, exist_ok=True)
        sess.run(tf.global_variables_initializer())
        print("Model initialized")
        
        os.makedirs(FLAGS.log_dir, exist_ok=True)
        os.makedirs(FLAGS.model_dir, exist_ok=True)
        FLAGS.model_dir = "{}/dcpo_dqn_model_{}".format(FLAGS.model_dir, FLAGS.suffix)
        os.makedirs(FLAGS.model_dir, exist_ok=True)
        summary_writer = tf.summary.FileWriter("logfiles/dcpo_dqn_summaries_{}".format(FLAGS.suffix), sess.graph)
        log_file = open("{}/dcpo_dqn_log_{}.txt".format(FLAGS.log_dir, FLAGS.suffix), "w")
        
        loss_training = 0
        # Initialize states at plan 0 for all agents
        s = np.zeros(NUM_AGENTS)
        # epsilon greedy parameter
        epsilon = FLAGS.epsilon
        var_prev = None
        var_min = None
        for episode in range(FLAGS.max_episodes):
            # Get the current Q values and actions
            Q_curr, a = sess.run([Q, actions_indices], feed_dict={states: s})
            # epsilon-greedy: perform random actions with epsilon probability
            if np.random.rand(1) < epsilon:
                # Decrease epsilon
                if episode < 3999:
                    epsilon = epsilon - 1/1000
                a = np.random.randint(0, NUM_ACTIONS, NUM_AGENTS)
            # Each action is representing the next states
            s_next = a
            # Compute the total variance of the next states
            s_next_timesteps = [dataset_input[i, s_next[i], :] for i in range(NUM_AGENTS)]
            var = np.var(np.sum(s_next_timesteps, axis=0), axis=0, ddof=1, dtype=np.float32)
            # Compute the reward as the variance of the current states minus the variance of the next states
            var_prev = var_prev or 2*var
            var_min = var_min or var
            if var < var_min:
                var_min = var
            reward = var_prev - var
            var_prev = var
            # Get the next Q values using the next states
            Q_next = sess.run(Q, feed_dict={states: s_next}) 
            # Calculate the maximum next Q for each agent and set our target value for the chosen actions
            max_Q_next = np.max(Q_next, axis=1)
            Q_targ = Q_curr
            for i in range(NUM_AGENTS):
                Q_targ[i, a[i]] = reward + FLAGS.gamma*max_Q_next[i]
            # Train our network using target and predicted Q values
            _, l, summary = sess.run([optimizer, loss, summary_merged],feed_dict={states:s ,Q_target:Q_targ, reward_input: reward, variance_input: var})
            # Update the states
            s = s_next
            # Accumulate the loss
            loss_training += l
            # Summaries
            summary_writer.add_summary(summary, episode)
            if episode%FLAGS.recording_episodes == 0:
                loss_training = loss_training/FLAGS.recording_episodes
                print_and_log("episode: {}, training loss: {:.5f}, reward {:.5f}, variance: {:.5f}, actions: {}".format(episode, loss_training, reward, var, a), log_file)
                loss_training = 0
                # Save the model
                save_path = saver.save(sess, FLAGS.model_dir + "/model.ckpt")
        
        print_and_log(str(var_min), log_file)
        log_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, default="", 
                        help="file containing the dataset")
    parser.add_argument("--model_dir", type=str, default="models", 
                        help="directory containing the model")
    parser.add_argument("--log_dir", type=str, default="logfiles", 
                        help="directory containing the logfiles and summaries")
    parser.add_argument("--max_episodes", type=int, default=100000,
                        help="maximum number of episodes for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, 
                        help="learning rate for the optimizer")
    parser.add_argument("--recording_episodes", type=int, default=250,
                        help="number of episodes trained before recording metrics and saving the model")
    parser.add_argument("--suffix", type=str, default="", 
                        help="suffix for the created files and directories")
    parser.add_argument("--epsilon", type=float, default=3.0, 
                        help="epsilon-greedy parameter")
    parser.add_argument("--gamma", type=float, default=0.1, 
                        help="discount factor for Q-learning")
    parser.add_argument("--arch", type=str, default="lstm", choices=["lstm", "cnn", "fc"], 
                        help="Q-network architecture to use")
    FLAGS = parser.parse_args()
    
    tf.app.run()
