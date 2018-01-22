import numpy as np
import math

import reinforced_epos.helpers.rl_helper_np as nrlh
from reinforced_epos.helpers.oop.helper_classes import TransitionType, BoundaryTransition
import threading

class Environment():
    bvar = None
    def __init__(self, dataset, transition:int, boundary_transition:int, total_moves=3):
        self.dataset = dataset
        self.total_moves = total_moves
        self.transition = transition
        self.boundary_transition = boundary_transition
        shape = np.shape(dataset)
        self.total_agents = shape[0]
        self.total_plans = shape[1]
        self.total_timesteps = shape[2]
        self.best_var = None
        self.success = False
        self.current_var = None

        if transition == TransitionType.ADDITION:
            self.total_actions = total_moves
        else:
            self.total_actions = self.total_plans

        self.current_state = nrlh.random_state(self.total_plans, self.total_agents)
        self.previous_current_state = np.copy(self.current_state)

    def transit_state(self, state, agent_actions):
        action_vector = agent_actions
        #print(self.transition)
        if self.transition == TransitionType.ADDITION:
            action_vector = agent_actions - self.total_actions//2 #create a movement
        #print(self.transition)
        if self.transition == TransitionType.ADDITION:
            raw_state = np.add(state, action_vector)
        elif TransitionType.ASSIGNMENT:
            raw_state = action_vector

        if self.boundary_transition == BoundaryTransition.MOD:
            return np.mod(raw_state, self.total_plans)
        elif self.boundary_transition == BoundaryTransition.CLIP:
            return np.clip(raw_state, 0, self.total_plans-1)

    def get_plans_from_state(self, state):
        tensors = [self.dataset[i, state[i], :] for i in range(state.shape[0])]
        #print(np.shape(tensors))
        return np.stack(tensors, axis=0)

    def state_variance(self, state, update=True):
        shape = np.shape(self.dataset)
        sum_a = np.zeros(shape[2])
        for i in range(shape[0]):
            sum_a = np.add(sum_a, self.dataset[i, state[i], :])
        var = np.var(sum_a, ddof=1)
        self.var = var
        prev_best_var = None
        success = False
        if update ==  True:
            lock = threading.RLock()
            lock.acquire()
            self.best_var = Environment.bvar
            self.best_var = self.best_var or var
            if(var < self.best_var):
                print(var)
                success = var < self.best_var
            self.best_var = var if var < self.best_var else self.best_var
            if Environment.bvar is not None:
                prev_best_var = float(Environment.bvar)
            else:
                prev_best_var = var
            Environment.bvar = self.best_var
            lock.release()
        return var, prev_best_var, success


    def is_episode_finished(self):
        return self.best_var == self.current_var and self.current_var is not None and self.best_var is not None

    def step(self, agent_actions):
        new_state = self.transit_state(self.current_state, agent_actions)
        var, prev_best, success = self.state_variance(new_state)
        return new_state, var, prev_best, success


    def total_actions(self):
        return self.total_actions

    def reset(self, random = True):
        if(random):
            self.current_state = nrlh.random_state(self.total_plans, self.total_agents)
            self.previous_current_state = np.copy(self.current_state)
        else:
            self.current_state = self.previous_current_state
        return self.current_state

