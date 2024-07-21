#!/usr/bin/env python3
import numpy as np
import filterpy
class ParticleFilter:
    def __init__(self, num_states, init_distribution, motion_model, obs_model, num_particles, min_num_effective_particles=1.5):
        self._particles = np.zeros((num_particles, num_states))
        self._weights = np.ones((num_particles))/float(num_particles)
        self._min_num_effective_particles = min_num_effective_particles

        self._init_distribution = init_distribution
        self._motion_model = motion_model
        self._obs_model = obs_model

    def norm_weights(self):
        self._weights = self._weights/np.sum(self._weights)

    def init_filter(self, std):
        num_particles = self._particles.shape[0]
        tiled_std = np.tile(std, (num_particles, 1))
        self._particles = self._init_distribution(tiled_std)

    def predict(self, std):
        num_particles = self._particles.shape[0]
        tiled_std = np.tile(std, (num_particles, 1))
        self._particles = self._motion_model(self._particles, tiled_std)


    def update(self, points_2d, ctrnet, joint_angles, cam, cTr, gamma):
        num_particles = self._particles.shape[0]
        obs_probs = self._obs_model(self._particles, points_2d, ctrnet, joint_angles, cam, cTr, gamma)
        self._weights = self._weights*obs_probs[:, 0]
        self.norm_weights()
        # for p_idx, particle in enumerate(self._particles):
        #      obs_prob = self._obs_model(particle, points_2d, ctrnet, joint_angles, cam, cTr, gamma)
        #      self._weights[p_idx] = self._weights[p_idx]*obs_prob
        # print(self._weights)
        # self.norm_weights()

        # TODO: resample if particles are depleted
        thresh = 1./np.sum(self._weights**2)
        # print(thresh)
        if self._min_num_effective_particles > thresh or np.isnan(thresh):
            indices = filterpy.monte_carlo.stratified_resample(self._weights)
            self._particles = self._particles[indices, :]
            self._weights = np.ones((num_particles))/float(num_particles)

    def resample(self):
        pass


    def get_mean_particle(self):
        self.norm_weights()
        return np.dot(self._weights, self._particles)