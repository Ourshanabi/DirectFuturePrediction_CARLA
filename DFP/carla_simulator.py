from __future__ import print_function
import sys
import os

import random
import time
import numpy as np
import re
import cv2
import argparse
import collections
import datetime
import glob
import logging
import math
import weakref

import numpy as np

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================
# CARLA_PATH = "/home/marius/carla/carla2/"

# try:
#     sys.path.append(glob.glob(CARLA_PATH + "PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg")[0])
# except IndexError:
#     pass

# ==============================================================================
# -- add PythonAPI for release mode --------------------------------------------
# ==============================================================================
# try:
#     sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
# except IndexError:
#     pass

import carla

class CarlaSimulator:

    def __init__(self, args):
        self.resolution = args['resolution']
        self.frame_skip = args['frame_skip']
        self.color_mode = args['color_mode']
        self.switch_maps = args['switch_maps']
        self.maps = args['maps']
        self.game_args = args['game_args']
        
        self.num_ped = 10
                
        self.available_controls, self.continuous_controls, self.continuous_range, self.discrete_controls = self.analyze_controls()  
        self.num_buttons = len(self.available_controls)
        assert(self.num_buttons == len(self.discrete_controls) + len(self.continuous_controls))
        assert(len(self.continuous_controls) == 0) # only discrete for now
        
        self.client = carla.Client(args['host'], args['port'])
        self.client.set_timeout(10.0)

        self.world = self.client.get_world()

        #Synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        self.world.apply_settings(settings)

        self.blueprint_library = self.world.get_blueprint_library()

        self.vehicle_blueprints = self.blueprint_library.filter('vehicle.*')
        self.pedestrians_blueprints = self.blueprint_library.filter('walker.pedestrian.*')

        self.spawn_points = self.world.get_map().get_spawn_points()

        self.raw_img = None
        self.raw_depth = None
    
    def update_img(self, img):
        self.raw_img = img

    def update_depth(self, depth):
        self.raw_depth = depth

    def analyze_controls(self):
        avail_controls = ['THROTTLE', 'BRAKE', 'LEFT', 'RIGHT']
        cont_controls = np.array([])
        cont_ranges = np.array([])
        discr_controls = np.array([0,1,2,3])
        return avail_controls, cont_controls, cont_ranges, discr_controls

    def init_game(self):
        pass

    def close_game(self):
        pass

    def step(self, action=0):
        """
        Action can be either the number of action or the actual list defining the action
        
        Args:
            action - action encoded either as an int (index of the action) or as a bool vector
        Returns:
            img  - image after the step
            meas - numpy array of returned additional measurements (e.g. health, ammo) after the step
            rwrd - reward after the step
            term - if the state after the step is terminal
        """
        pass

    def get_random_action(self):
        action = [None] * self.num_action
        for i in self.discrete_controls:
            action[i] = (random.random() >= .5)
        for i, (min, max) in zip(self.continuous_controls, self.continuous_range):
            action[i] = min + random.random() * (max - min)
        return action

    def is_new_episode(self):
        pass
        
    def next_map(self):
        print("\n\n ## Next map not implemented for Carla ## \n\n")
    
    def new_episode(self):
        raise NotImplementedError

