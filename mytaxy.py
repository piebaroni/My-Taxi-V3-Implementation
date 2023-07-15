from contextlib import closing
from io import StringIO
from os import path
from typing import Optional
import random

import numpy as np

from gym import Env, logger, spaces, utils
from gym.envs.toy_text.utils import categorical_sample
from gym.error import DependencyNotInstalled

MAP = [
    "+---------+",
    "|G: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|G| : |G: |",
    "+---------+",
]
(0, 0), (0, 4), (4, 0), (4, 3)

free_loc = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), 
            (3, 0), (3, 1), (3, 2), (3, 3), (3, 4)]

class TaxiEnv2(Env):
    """
    Map:

        +---------+
        |R: | : :G|
        | : | : : |
        | : : : : |
        | | : | : |
        |Y| : |B: |
        +---------+

    ### Actions
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: drop off passenger
    - 6: make gas

    ### Observations
    There are 5500 discrete states since there are 25 taxi positions, 5 possible
    locations of the passenger (including the case when the passenger is in the
    taxi), 4 destination locations and 11 levels of gas.

    Each state space is represented by the tuple:
    (taxi_row, taxi_col, passenger_location, destination, gas_tank)

    An observation is an integer that encodes the corresponding state.
    The state tuple can then be decoded with the "decode" method.

    Passenger locations:
    - 0: R(red) 
    - 1: G(green) 
    - 2: Y(yellow) 
    - 3: B(blue) 
    - 4: in taxi

    Destinations:
    - 0: R(red) 
    - 1: G(green) 
    - 2: Y(yellow) 
    - 3: B(blue) 

    Petrol Stations Locations:
    - 0: P casual
    - 1: P casual
    - 2: P casual
    - 3: P casual
    - 4: P casual (position 1 from taxi spawn)

    ### Rewards
    - -1 per step unless other reward is triggered.
    - +20 delivering passenger.
    - -10 executing "pickup", "drop-off", "make gas" actions illegally.
    - -10 terminating gas
    """

    def __init__(self, render_mode: Optional[str] = None):
        self.desc = np.asarray(MAP, dtype="c")
        self.locs = [(0, 0), (0, 4), (4, 0), (4, 3)]
        self.locs_colors = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255)]
        self.reset()
        
    def encode(self, taxi_row, taxi_col, pass_loc, dest_idx, gas_tank):
        # (5) 5, 5, 4, 11
        i = taxi_row
        i *= 5
        i += taxi_col
        i *= 5
        i += pass_loc
        i *= 4
        i += dest_idx
        i *= 11
        i += gas_tank
        return i

    def decode(self, i):
        out = []
        out.append((i % 11))
        i = i // 11
        out.append(i % 4)
        i = i // 4
        out.append(i % 5)
        i = i // 5
        out.append(i % 5)
        i = i // 5
        out.append(i)
        assert 0 <= i < 5
        return reversed(out)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        #Set new Taxi position
        self.s = -1
        passengers = 4
        arrival = 3
        row = 0
        col = 0
        gas = 0
        possible_positions = []
        while(self.s == -1 or self.s > 5499):
            row = random.randint(0, 4)  #random position
            col = random.randint(0, 4)  #random position
            gas = random.randint(1, 10) #random gas tank
            self.s = self.encode(row, col, passengers, arrival, gas)
        self.lastaction = None
        self.taxi_orientation = 0
        #print(self.s)
        # Set petrol station positions
        self.petrol = random.sample(free_loc, 4)
        
        # Choose petrol station near the taxi spawn
        possible_positions = [(row + dr, col + dc)
                              for dr in [-1, 0, 1] for dc in [-1, 0, 1]
                              if 0 <= row + dr < 5 and 0 <= col + dc < 5
                              and (dr == 0) != (dc == 0) 
                              and (dr, dc) not in self.locs
                              and (dr, dc not in self.petrol)]
        
        new_element_position = random.choice(possible_positions)
        self.petrol.append(new_element_position)
        # print(self.locs)
        # print(self.petrol)
        # print(list(self.decode(self.s)))
        num_states = 5500
        num_rows = 5
        num_columns = 5
        max_row = num_rows - 1
        max_col = num_columns - 1
        gas_tank = 11
        num_actions = 7
        self.P = {
            state: {action: [] for action in range(num_actions)}
            for state in range(num_states)
        }
        for row in range(num_rows):
            for col in range(num_columns):
                for pass_idx in range(len(self.locs) + 1):  # +1 for being inside taxi
                    for dest_idx in range(len(self.locs)):
                        for gas_i in range(gas_tank):
                            state = self.encode(row, col, pass_idx, dest_idx, gas_i)
                            terminated = False
                            taxi_loc = (row, col)
                            reward = (
                                -1
                            ) # default reward when there is no pickup/dropoff
                            for action in range(num_actions):
                                new_row, new_col, new_pass_idx = row, col, pass_idx
                                if gas_i == 0:
                                    terminated = True
                                    new_tank = 0
                                    reward = -10
                                else:
                                    if action == 0:
                                        new_row = min(row + 1, max_row)
                                        new_tank = max(0, gas_i - 1)
                                    elif action == 1:
                                        new_row = max(row - 1, 0)
                                        new_tank = max(0, gas_i - 1)
                                    elif action == 2 and self.desc[1 + row, 2 * col + 2] == b":":
                                        new_col = min(col + 1, max_col)
                                        new_tank = max(0, gas_i - 1)
                                    elif action == 3 and self.desc[1 + row, 2 * col] == b":":
                                        new_col = max(col - 1, 0)
                                        new_tank = max(0, gas_i - 1)
                                    elif action == 4:  # pickup
                                        if pass_idx < 4 and taxi_loc == self.locs[pass_idx]:
                                            new_pass_idx = 4
                                        else:  # passenger not at location
                                            reward = -10
                                    elif action == 5:  # dropoff
                                        if (taxi_loc == self.locs[dest_idx]) and pass_idx == 4:
                                            new_pass_idx = dest_idx
                                            terminated = True
                                            reward = 20
                                        elif (taxi_loc in self.locs) and pass_idx == 4:
                                            new_pass_idx = self.locs.index(taxi_loc)
                                        else:  # dropoff at wrong location
                                            reward = -10
                                    elif action == 6:  # make gas
                                        if (row, col) in self.petrol: # MAKE GAS
                                            new_tank = 10
                                            #reward = 20
                                        else:
                                            new_tank = max(0, gas_i - 1)
                                            reward = -10
                    
                                new_state = self.encode(
                                    new_row, new_col, new_pass_idx, dest_idx, new_tank)

                                self.P[state][action].append(
                                    (1.0, new_state, reward, terminated))
                        
        # self.initial_state_distrib /= self.initial_state_distrib.sum()
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Discrete(num_states)
        #print(self.P)
        return int(self.s), {"prob": 1.0, "action_mask": self.action_mask(self.s)}
    
    def action_mask(self, state: int):
        """Computes an action mask for the action space using the state information."""
        mask = np.zeros(7, dtype=np.int8)
        taxi_row, taxi_col, pass_loc, dest_idx, gas_tank = self.decode(state)
        if taxi_row < 4 and gas_tank != 0:
            mask[0] = 1
        if taxi_row > 0 and gas_tank != 0:
            mask[1] = 1
        if taxi_col < 4 and self.desc[taxi_row + 1, 2 * taxi_col + 2] == b":" and gas_tank != 0:
            mask[2] = 1
        if taxi_col > 0 and self.desc[taxi_row + 1, 2 * taxi_col] == b":" and gas_tank != 0:
            mask[3] = 1
        if pass_loc < 4 and (taxi_row, taxi_col) == self.locs[pass_loc] and gas_tank != 0:
            mask[4] = 1
        if pass_loc == 4 and (
            (taxi_row, taxi_col) == self.locs[dest_idx]
            or (taxi_row, taxi_col) in self.locs
        ) and gas_tank != 0:
            mask[5] = 1
        if (taxi_row, taxi_col) in self.petrol:
            mask[6] = 1
        return mask

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        self.s = s
        self.lastaction = a
        return (int(s), r, t,{"prob": p, "action_mask": self.action_mask(s)})


    def mappa(self):
        return(self.desc)