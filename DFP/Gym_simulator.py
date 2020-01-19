'''
ViZDoom wrapper
'''
from __future__ import print_function
import sys
import os

vizdoom_path = 'C://Users//Rzhang//Anaconda3//envs//recognition//Lib//site-packages//vizdoom'
sys.path = [os.path.join(vizdoom_path,'bin/python3')] + sys.path

import vizdoom 
print(vizdoom.__file__)
import random
import time
import numpy as np
import re
import cv2
import gym



def get_screen(env):
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)

class Gym_simulator:
    
    def __init__(self, args):        
        self.config = args['config']
        self.resolution = args['resolution']
        self.frame_skip = args['frame_skip']
        self.color_mode = args['color_mode']
        self.switch_maps = args['switch_maps']
        self.maps = args['maps']
        self.game_args = args['game_args']

        self.env_name = args['env_name']
        self.resolution = args['resolution']
        self.num_meas = args['num_meas']

        self._env = gym.make(self.env_name).unwrapped

        self.counter = 0



        # self._game = vizdoom.DoomGame()
        # self._game.set_vizdoom_path(os.path.join(vizdoom_path,'vizdoom'))
        # self._game.set_doom_game_path(os.path.join(vizdoom_path,'freedoom2.wad'))
        # self._game.load_config(self.config)
        # self._game.add_game_args(self.game_args)
        # self.curr_map = 0
        # self._game.set_doom_map(self.maps[self.curr_map])
        
        # # set resolution
        # try:
        #     self._game.set_screen_resolution(getattr(vizdoom.ScreenResolution, 'RES_%dX%d' % self.resolution))
        #     self.resize = False
        # except:
        #     print("Requested resolution not supported:", sys.exc_info()[0], ". Setting to 160x120 and resizing")
        #     self._game.set_screen_resolution(getattr(vizdoom.ScreenResolution, 'RES_160X120'))
        self.resize = True

        # set color mode
        if self.color_mode == 'RGB':
            self.num_channels = 3
        elif self.color_mode == 'GRAY':
            self.num_channels = 1
        else:
            print("Unknown color mode")
            raise

        self.available_controls, self.continuous_controls, self.discrete_controls = self.analyze_controls(self.config)
        self.num_buttons = self._env.action_space.n
        assert(self.num_buttons == len(self.discrete_controls) + len(self.continuous_controls))
        assert(len(self.continuous_controls) == 0) # only discrete for now
        

        #self.num_meas = self._game.get_available_game_variables_size()
            
        self.meas_tags = []
        for nm in range(self.num_meas):
            self.meas_tags.append('meas' + str(nm))
            
        self.episode_count = 0
        self.game_initialized = False

    def get_screen(self):
        screen = self._env.render(mode='rgb_array')
        
        return screen

    def analyze_controls(self, config_file):
        with open(config_file, 'r') as myfile:
            config = myfile.read()
        m = re.search('available_buttons[\s]*\=[\s]*\{([^\}]*)\}', config)
        avail_controls = m.group(1).split()
        cont_controls = np.array([bool(re.match('.*_DELTA', c)) for c in avail_controls])
        discr_controls = np.invert(cont_controls)
        return avail_controls, np.squeeze(np.nonzero(cont_controls)), np.squeeze(np.nonzero(discr_controls))
        
    def init_game(self):
        if not self.game_initialized:
            self._env.reset()
            self.game_initialized = True
            self.counter = 0
            
    def close_game(self):
        if self.game_initialized:
            self.game_initialized = False
            
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

        state, rwrd, done, _ = self._env.step(np.argmax(action))
        self.counter += 1
        
        
        if state is None:
            img = None
            meas = None
        else:        
            
            if self.color_mode == 'RGB':
                raw_img = self.get_screen()[None,:,:,:]
            elif self.color_mode == 'GRAY':
                raw_img = np.expand_dims(state.screen_buffer,0)
                
            if self.resize:
                    if raw_img is None or (isinstance(raw_img, list) and raw_img[0] is None):
                        img = None
                    else:
                        img = cv2.resize(raw_img[0], (self.resolution[0], self.resolution[1])).transpose((2, 0, 1))
            else:
                img = raw_img
                
            meas = state #[[0,2]] # will decide later what is a good measurement for each env

        term = done
        
        if term:
            if self.env_name == 'CartPole-v1':
                if self.counter > 200:
                    print("EPISODE DONE IN")
                    print(self.counter)
            else : 
                print("EPISODE DONE IN")
                print(self.counter)
            self.new_episode() # in multiplayer multi_simulator takes care of this            
            img = np.zeros((self.num_channels, self.resolution[1], self.resolution[0]), dtype=np.uint8) # should ideally put nan here, but since it's an int...
            meas = np.zeros(self.num_meas, dtype=np.uint32) # should ideally put nan here, but since it's an int...


        return img, meas, rwrd, term
    
    def get_random_action(self):
        return [(random.random() >= .5) for i in range(self.num_buttons)]
        
    def is_new_episode(self):
        return self._game.is_new_episode()
    
    def new_episode(self):
        self.episode_count += 1
        self.counter = 0
        self._env.reset()