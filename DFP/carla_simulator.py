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
        self.config = args['config']
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

        if args['color_mode'] == 'RGB':
            self.num_channels = 3
        else :
            raise ValueError("RGB is the only mode implemented")

        # measures
        self.num_meas = 14 # pos 6D, vel 3D, acc 3D, collision 1D, lane invasion 1D


        #Synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        self.world.apply_settings(settings)
        self.world.tick()

        self.blueprint_library = self.world.get_blueprint_library()

        self.vehicle_blueprints = self.blueprint_library.filter('vehicle.*')
        self.pedestrians_blueprints = self.blueprint_library.filter('walker.pedestrian.*')

        self.spawn_points = self.world.get_map().get_spawn_points()

        self.initialized = False

        self.raw_img = None
        self.raw_depth = None
        self.collided = False
        self.lane_crossed = False
    
    def update_img(self, img):
        W, H = img.width, img.height
        img_np = np.asarray(img.raw_data)
        self.raw_img = img_np.reshape(H,W,4)[:,:,:3]
        print("pif")

    def update_depth(self, depth):
        # imnp_ss = np.asarray(image_ss.raw_data)
        # imnp2_ss = imnp_ss.reshape(600,800,4)[:,:,2]
        # self.raw_depth = depth
        raise NotImplementedError("RGBD mode not implemented")

    def collision_handler(self, event):
        self.collided = True

    def lane_handler(self, event):
        self.lane_crossed = True


    def analyze_controls(self):
        avail_controls = ['THROTTLE', 'BRAKE', 'LEFT', 'RIGHT']
        cont_controls = np.array([])
        cont_ranges = np.array([])
        discr_controls = np.array([0,1,2,3])
        return avail_controls, cont_controls, cont_ranges, discr_controls

    def init_game(self):
        if not self.initialized:
            vehicle_bp = random.choice(self.vehicle_blueprints)
            transform = random.choice(self.spawn_points)
            self.actor = self.world.spawn_actor(vehicle_bp, transform)

            camera_bp = self.blueprint_library.find('sensor.camera.rgb')
            relative_transform = carla.Transform(carla.Location(x=0, y=0, z=1.5))
            self.camera = self.world.spawn_actor(camera_bp, relative_transform, attach_to = self.actor)
            self.camera.image_size_x, self.camera.image_size_y = self.resolution
            self.camera.listen(self.update_img)

            if self.color_mode == 'RGBD':
                camera_ss_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
                self.camera_ss = world.spawn_actor(camera_ss_bp, relative_transform, attach_to = self.actor)
                self.camera_ss.image_size_x, self.camera_ss.image_size_y = self.resolution
                self.camera_ss.listen(self.update_depth)
            
            collision_sensor_bp = self.blueprint_library.find('sensor.other.collision')
            self.collision_sensor = self.world.spawn_actor(collision_sensor_bp,
                                        carla.Transform(), attach_to= self.actor)
            self.collision_sensor.listen(self.collision_handler)

            lane_cross_sensor_bp = self.blueprint_library.find('sensor.other.lane_invasion')
            self.crossing_sensor = self.world.spawn_actor(lane_cross_sensor_bp,
                                        carla.Transform(), attach_to= self.actor)
            self.crossing_sensor.listen(self.lane_handler)

            self.world.tick()

            self.initialized = True

    def close_game(self):
        if self.initialized:
            self.actor.destroy()
            self.camera.destroy()
            if self.color_mode == 'RGBD':
                self.camera_ss.destroy()
            self.crossing_sensor.destroy()
            self.collision_sensor.destroy()
            self.initialized = False

    def apply_action(self, action):
        control = carla.VehicleControl(
            throttle = action[0],
            steer = action[3] - action[2],
            brake = action[1],
            hand_brake = False,
            reverse = False,
            manual_gear_shift = False,
            gear = 0)
        self.actor.apply_control(control)

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
        self.init_game()
        
        self.apply_action(action)
        self.collided = False
        self.lane_crossed = False
        print("start tick")
        self.world.tick()
        print("end tick")

        raw_img = self.raw_img
        if self.color_mode == 'RGBD':
            depth = self.raw_depth
            raw_img = np.vstack((raw_img, depth))
        
        transform = self.actor.get_transform()
        pos = transform.location
        rot = transform.rotation
        vel = self.actor.get_velocity()
        acc = self.actor.get_acceleration()
        
        meas = np.zeros(self.num_meas)
        meas[:3] = pos.x, pos.y, pos.z
        meas[3:6] = rot.pitch, rot.yaw, rot.roll
        meas[6:9] = vel.x, vel.y, vel.z
        meas[9:12] = acc.x, acc.y, acc.z
        meas[12] = self.collided
        meas[13] = self.lane_crossed

        term = self.collided

        speed = (vel.x**2 + vel.y**2 + vel.z**2)**0.5

        target_speed = 5

        reward = speed * (speed - 2 * target_speed) - 2 * self.collided

        if term:
            self.new_episode()
        
        return raw_img, meas, reward, term

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
        self.close_game()


