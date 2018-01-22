import numpy as np
import tensorflow as tf

from reinforced_epos.helpers.oop.helper_classes import TransitionType, BoundaryTransition


#supper nightly build

#state a collection of plan indexes of shape [users]
#action a collection of possible actions, which are integers added to the current state
def random_state(total_plans, total_users):
    return tf.random_uniform([total_users],
        minval=0, maxval=total_plans, dtype=tf.int32, name="random_plan_selection")

def random_action(actions, total_users):
    action_indeces = tf.random_uniform([total_users],
        minval=0, maxval=actions.shape[0], dtype=tf.int32, name="random_action_selection")

    return  tf.map_fn(elems=action_indeces, fn=lambda i: actions[i])

def get_plans(data, state):
    print(state.shape[0])
    tfs = [data[i, state[i], :] for i in range(0, state.shape[0])]
    return tf.stack(tfs, axis=0)

def state_transition(action, state, total_plans, bounray_transition: BoundaryTransition, transition: TransitionType):

    raw_state = tf.add(action, state, name="state_transition") if transition == TransitionType.ADDITION else action
    if bounray_transition == BoundaryTransition.MOD:
        return tf.mod(raw_state, [total_plans], name="state_mod")
    elif bounray_transition == BoundaryTransition.CLIP:
        return tf.clip_by_value(raw_state, 0, total_plans, name="state_clip")

def reward(data, state):
    summed_plan = tf.reduce_sum(get_plans(data, state), axis=0)
    # numerical instability detected in tf.momments method
    return np.var(summed_plan.eval(), axis=0, ddof=1, dtype=np.float64)

#qvalues should be a [users x total_actions] array
def arg_max_qvalues(qvalues):
    return tf.argmax(qvalues, axis=0, name="q_to_actions")

def dqn_loss(q, q_target, gamma_target, reward_target):
    return tf.pow(tf.subtract(tf.add(reward_target, tf.multiply(gamma_target, q_target)), q), 2)

if __name__ == "__main__":
    users = 4
    total_plans = 7
    timesteps = 11
    actions = tf.constant(value=[-1, 0, 1], shape=[3], dtype=tf.int32)
    total_actions  = actions.shape[0]

    shape = [users, total_plans, timesteps]
    action_shape = [users, 3]
    sess = tf.Session()
    dataset = tf.get_variable("my_int_variable", shape, dtype=tf.float32,
                           initializer=tf.uniform_unit_scaling_initializer)
    ovar = tf.get_variable("output_var", shape )

    with sess.as_default():
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        init_state = random_state(total_plans, users).eval()

        print(init_state)

        plans = get_plans(dataset, init_state).eval()
        #print(plans)
        action = random_action(actions, users).eval()
        print(action)
        new_state = state_transition(action=action, state=init_state, total_plans=total_plans)
        print(new_state.eval())
        print(reward(dataset, new_state))
        print(reward(dataset, new_state))

        #qvalues = tf.random_uniform(shape=[users, actions.shape[0]],dtype=tf.float32, minval=0),eval()
        #bestQ = arg_max_qvalues(qvalues=qvalues).eval()
        #print(qvalues)
