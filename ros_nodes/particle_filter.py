#!/usr/bin/env python3
import numpy as np

class ParticleFilter:
    def __init__(self, init_distribution, motion_model, obs_model, num_particles=1000, min_num_effective_particles=0.5):
        self._particles_r = np.zeros((num_particles, 1, 4))
        self._particles_t = np.zeros((num_particles, 3))
        self._num_particles
        self._min_num_effective_particles = min_num_effective_particles

        self._init_distribution = init_distribution
        self._motion_model = motion_model
        self._obs_model = obs_model

    def init_filter(self, std_r, std_t):
        self._omega_r = np.random.normal(0, std_r, size=(self._num_particles, 1, 4))
        self._omega_t = np.random.normal(0, std_t, size=(self._num_particles, 3))

    def predict(self):
        

    def update(self):
        pass

    def resample(self):
        pass

    def get_max_particles(self):
        pass