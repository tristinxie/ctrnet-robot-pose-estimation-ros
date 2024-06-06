#!/usr/bin/env python3
import numpy as np

class ParticleFilter:
    def __init__(self, num_states, init_distribution, motion_model, obs_model, num_particles=1000, min_num_effective_particles=0.5):
        self._particles = np.zeros((num_particles, ))
        self._weights = np.ones((num_particles))/float(num_particles)

        self._init_distribution = init_distribution
        self._motion_model = motion_model
        self._obs_model = obs_model

    def norm_weights(self):
        self._weights = self._weights/np.sum(self._weights)

    def init_filter(self, std_r, std_t):
        # self._omega_r = np.random.normal(0, std_r, size=(self._num_particles, 1, 4))
        # self._omega_t = np.random.normal(0, std_t, size=(self._num_particles, 3))

        for p_idx, _ in enumerate(self._particle):
             self._particles[p_idx], _ = self._init_distribution(self._std)

    def predict(self):
        for p_idx, particle in enumerate(self._particles):
                self._particles[p_idx], _ = self._motion_model(particle, self._std)


    def update(self, points_2d, ctrnet, joint_angles, cam, cTr, gamma):
        for p_idx, particle in enumerate(self._particles):
             obs_prob = self._obs_model(particle, points_2d, ctrnet, joint_angles, cam, cTr, gamma)
             self._weights[p_idx] = self._weights[p_idx]*obs_prob
        
        self.norm_weights()

        # TODO: resample if particles are depleted

    def resample(self):
        pass

    def get_mean_particle(self):
        self.norm_weights()
        return np.dot(self._weights, self._particles)