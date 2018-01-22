from builtins import print
import os
from reinforced_epos.helpers.oop.Network import  Network as AC_Network
import tensorflow as tf
from reinforced_epos.helpers.oop.helpers import *
from reinforced_epos.helpers.oop.Environment import Environment
from reinforced_epos.helpers.config import get_experiement_folder
import pickle
import shutil

class Worker():
    def __init__(self,name, dataset, env: Environment, total_actions, trainer, model_path, global_episodes):
        self.name = "worker_" + str(name)
        self.number = name
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        folder = get_experiement_folder()
        a3cfolder = os.path.join(folder, "a3c")
        self.model_path = a3cfolder


        worker_folder = os.path.join(folder, "a3c", "train_"+str(self.number))

        self.summary_writer = tf.summary.FileWriter(worker_folder)
        self.successful_episodes_path = os.path.join(worker_folder, "successfull_episodes")
        if(not os.path.exists(self.successful_episodes_path)):
            os.mkdir(self.successful_episodes_path)

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(dataset, total_actions, self.name, trainer)
        self.update_local_ops = update_target_graph('global',self.name)

        self.actions = None #here i did a major change that I don't know how it will work
        self.rewards_plus = None
        self.value_plus = None
        self.env = env
        # actions are used in the AC_network to determine the q-values, so they should be
        # agent x actions matrix


    #rollout became episode_memories, bootstrap is either 0.0 or a value from a feedforward of the net
    def train(self, episode_memories, sess, gamma, bootstrap_value):
        #print("training")
        episode_memories = np.array(episode_memories)
        observed_states = episode_memories[:, 0] #instead of observations, batch x agent x states
        actions = episode_memories[:, 1] # batch x agent x acions
        rewards = episode_memories[:, 2] # batch x 1
        next_observations = episode_memories[:, 3] # unused?
        values = episode_memories[:, 5] # batch x 1

        # Here we take the rewards and values from the episode_memories, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value]) #increase all rewards with bootstrap
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1] #discount all rewards except the last
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value]) #also adding bootstrap to values
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1] #advantages are calculated
        # as the sum of the rewards plus the discounted values (after first) minus all the values except the last
        advantages = discount(advantages, gamma) #batch x 1


        #TODO solve dim mess
        discounted_rewards = np.expand_dims(discounted_rewards, 1)
        advantages = np.expand_dims(advantages, 1)
        o_states = np.stack(observed_states, axis=0)
        actions = np.expand_dims(actions[0], 0)
        # print(actions)
        b1 = self.batch_rnn_state[0]
        b2 = self.batch_rnn_state[1]

        # print(np.shape(b1))

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {
            self.local_AC.target_v:discounted_rewards,
            self.local_AC.input:o_states, # a batch of observed states
            self.local_AC.actions:actions, # a batch of actions
            self.local_AC.advantages:advantages, # a batch of advantages
            self.local_AC.lstm_h_prev:b1[0],
            self.local_AC.lstm_c_prev:b2[0]
        }

        v_l,p_l,e_l,g_n,v_n, self.batch_rnn_state,_ = sess.run([self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.entropy,
            self.local_AC.grad_norms,
            self.local_AC.var_norms,
            self.local_AC.state_out,
            self.local_AC.apply_grads],
            feed_dict={
            self.local_AC.target_v:discounted_rewards,
            self.local_AC.input:o_states, # a batch of observed states
            self.local_AC.actions:actions, # a batch of actions
            self.local_AC.advantages:advantages, # a batch of advantages
            self.local_AC.lstm_h_prev:np.expand_dims(b1[0], 0),
            self.local_AC.lstm_c_prev:np.expand_dims(b2[0], 0)
        })
        return v_l / len(episode_memories), p_l / len(episode_memories), e_l / len(episode_memories), g_n, v_n

    def work(self,max_episode_length,gamma,sess,coord,saver):
        #print("working")
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print ("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_variances = []
                episode_step_count = 0
                d = False

                #self.env.new_episode()
                s_indeces = self.env.current_state #indeces
                episode_frames.append(s_indeces) #append indeces
                s = self.env.get_plans_from_state(s_indeces) #plans

                rnn_state = self.local_AC.state_init
                self.batch_rnn_state = rnn_state

                var_prev = None
                self.env.reset(random=True)
                while self.env.is_episode_finished() == False:
                    #Take an action using probabilities from policy network output.
                    a_dist,v,rnn_state = sess.run([self.local_AC.policy,self.local_AC.value,self.local_AC.state_out],
                        feed_dict={self.local_AC.input:[s],
                        self.local_AC.state_in[0]:rnn_state[0],
                        self.local_AC.state_in[1]:rnn_state[1]})


                    agents = np.shape(self.env.dataset)[0]
                    a = np.zeros([agents],  dtype=np.int)
                    indeces = np.arange(self.env.total_actions)
                    for agent in range(agents):
                        a[agent] = np.random.choice(indeces, p=a_dist[0, agent, :])

                    # a = np.random.choice(a_dist[0],p=a_dist[0]) #choose the action based on the
                    #probability density of the policy gradient
                    # a = np.argmax(a_dist[0] == a, axis=1) # get indeces from probabilities
                    # print(a)
                    new_state, var, prev_best, d = self.env.step(a) #calcualtes new state

                    var_prev = var_prev or var
                    # print(var)

                    #r =  (self.env.best_var - var)#/ 100.0#self.actions[a])
                    r =  (prev_best - var)
                    episode_variances.append(var)
                    var_prev = var

                    # d = self.env.is_episode_finished()

                    if d == False:
                        s1_indeces = self.env.current_state
                        episode_frames.append(s1_indeces)
                        s1 = self.env.get_plans_from_state(s1_indeces)
                    else:
                        s1 = s

                    episode_buffer.append([s, a, r, s1, d, v])
                    episode_values.append(v)#v[0,0]

                    episode_reward += r
                    s = s1
                    total_steps += 1
                    episode_step_count += 1

                    if(episode_step_count > max_episode_length):
                        break;

                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if len(episode_buffer) == 30 and d != True and episode_step_count < max_episode_length - 1:
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        v1 = sess.run(self.local_AC.value,
                            feed_dict={self.local_AC.input:[s],
                            self.local_AC.state_in[0]:rnn_state[0],
                            self.local_AC.state_in[1]:rnn_state[1]})
                        v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    if d == True:
                        print("success")
                        file = os.path.join(self.successful_episodes_path, "episode_"+str(episode_count))
                        if(os.path.exists(file)):
                            os.remove(file)
                        with open(file, "wb+") as outfile:
                            pickle.dump(episode_buffer, outfile)
                            outfile.close()

                        break

                print(self.name + " " + "finished episode " + str(episode_count))
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # Update the network using the episode buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,0.0)


                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    if self.name == 'worker_0' and episode_count % 25 == 0:
                        state_transitions = np.array(episode_frames) #data_for_heatmap
                    if episode_count % 250 == 0 and self.name == 'worker_0':
                        saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
                        print ("Saved Model")

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    mean_var = np.mean(episode_variances)
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    summary.value.add(tag="Perf/Min Variance", simple_value=float(Environment.bvar))
                    summary.value.add(tag="Perf/Mean Variance", simple_value=float(mean_var))

                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1
