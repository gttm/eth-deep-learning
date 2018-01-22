import numpy as np

from reinforced_epos.helpers.oop.helper_classes import TransitionType, BoundaryTransition


def random_state(total_plans, total_users):
    return np.random.random_integers(0, total_plans-1, total_users)

def random_action(actions, total_users):
    return np.random.random_integers(0, actions, total_users)

def get_plans_from_state(data, state):
    tensors = [data[i, state[i], :] for i in range(0, state.shape[0])]
    return np.stack(tensors, axis=0)

def state_transition(action, state, total_plans, boundary_type: int, transition:int):
    print(transition)
    if transition == TransitionType.ADDITION:
        raw_state = np.add(action, state)
    elif TransitionType.ASSIGNMENT:
        raw_state = action

    if boundary_type == BoundaryTransition.MOD:
        return np.mod(raw_state, total_plans)
    elif boundary_type == BoundaryTransition.CLIP:
        return np.clip(raw_state, 0, total_plans-1)

def generate_q_target(q, q_new):
    q_target = np.copy(q)
    q_target[np.argmax(q_new, axis=1)] = np.max(q_new, axis=0)
    return q_target


def get_actions_from_q(q):
    return np.argmax(q, axis=1)

def reward(data, state):
    summed_plan = np.sum(get_plans_from_state(data, state), axis=0)
    return np.var(summed_plan, axis=0, ddof=1, dtype=np.float64)

def dqn_loss(q, q_target, reward_target, gamma_target):
    return np.power(np.subtract(np.add(reward_target, np.multiply(gamma_target, q_target)), q))
