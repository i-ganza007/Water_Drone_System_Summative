"""
custom_env.py
Core Wildfire Drone Coordination Environment

This file contains the environment logic without rendering code.
Rendering is handled separately in rendering.py for better modularity.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class WildfireDroneEnvironment(gym.Env):
    """
    Wildfire Fire-Fighting Drone Coordination Environment
    
    Problem: Autonomous wildfire response using AI-controlled drones
    Agent: Fire-fighting drone with water/retardant capacity
    Mission: Extinguish active fire hotspots before they spread
    Constraints: Limited water capacity, time-sensitive response
    
    Action Space: Discrete(5)
        0: Move North (up)
        1: Move South (down)
        2: Move West (left)
        3: Move East (right)
        4: Extinguish fire at current location
    
    Observation Space: Box(13,)
        [0]: Drone X position
        [1]: Drone Y position
        [2]: Water capacity remaining
        [3]: Number of fires extinguished
        [4-6]: Fire 1 (distance, dx, dy)
        [7-9]: Fire 2 (distance, dx, dy)
        [10-12]: Fire 3 (distance, dx, dy)
    
    Rewards:
        -0.1: Time step penalty
        +2.0/(1+dist): Reward for moving closer to fires
        -0.2: Wall collision penalty
        +200: Fire extinguished
        -0.5: Invalid extinguish attempt
        +500: All fires extinguished (mission success)
        +0.5*water: Fuel efficiency bonus
        -10: Ran out of water
        -5: Episode timeout
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, grid_size=10, num_fires=3, render_mode=None):
        super().__init__()
        self.grid_size = grid_size
        self.num_fires = num_fires
        self.render_mode = render_mode

        # Action space: Navigate (4 directions) + Extinguish
        self.action_space = spaces.Discrete(5)

        # Observation space
        obs_size = 4 + (num_fires * 3)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0] + [0, -grid_size, -grid_size] * num_fires, 
                        dtype=np.float32),
            high=np.array([grid_size, grid_size, 200, num_fires] + 
                         [np.sqrt(2*grid_size**2), grid_size, grid_size] * num_fires, 
                         dtype=np.float32),
            dtype=np.float32
        )

        # State variables
        self.drone_position = None
        self.fire_locations = None
        self.extinguished_fires = None
        self.water_capacity = 200
        self.max_water_capacity = 200
        self.steps = 0
        self.max_steps = 500
        
        # For rendering (will be set by rendering.py)
        self.renderer = None

    def reset(self, seed=None, options=None):
        """
        Reset environment to initial state
        
        Returns:
            observation: Initial observation
            info: Additional info dictionary
        """
        super().reset(seed=seed)
        
        # Random drone starting position
        self.drone_position = np.array(
            [self.np_random.integers(0, self.grid_size),
             self.np_random.integers(0, self.grid_size)],
            dtype=np.int32
        )
        
        # Generate fire hotspot locations
        self.fire_locations = []
        for _ in range(self.num_fires):
            while True:
                fire_pos = np.array(
                    [self.np_random.integers(0, self.grid_size),
                     self.np_random.integers(0, self.grid_size)],
                    dtype=np.int32
                )
                # Ensure fires don't spawn on drone or overlap
                if not np.array_equal(fire_pos, self.drone_position) and \
                   not any(np.array_equal(fire_pos, f) for f in self.fire_locations):
                    self.fire_locations.append(fire_pos)
                    break
        
        self.extinguished_fires = np.zeros(self.num_fires, dtype=bool)
        self.water_capacity = self.max_water_capacity
        self.steps = 0
        
        return self._get_observation(), {}

    def _get_observation(self):
        """
        Get current observation vector
        
        Returns:
            obs: Observation array containing drone position, water,
                 completed fires, and relative positions of all fires
        """
        obs = np.zeros(4 + self.num_fires * 3, dtype=np.float32)
        
        # Drone state
        obs[0] = float(self.drone_position[0])
        obs[1] = float(self.drone_position[1])
        obs[2] = float(self.water_capacity)
        obs[3] = float(np.sum(self.extinguished_fires))

        # Fire locations (relative to drone)
        fire_idx = 0
        for i in range(self.num_fires):
            if not self.extinguished_fires[i]:
                dx = float(self.fire_locations[i][0] - self.drone_position[0])
                dy = float(self.fire_locations[i][1] - self.drone_position[1])
                dist = np.sqrt(dx**2 + dy**2)
                obs[4 + fire_idx*3] = dist
                obs[4 + fire_idx*3 + 1] = dx
                obs[4 + fire_idx*3 + 2] = dy
            else:
                # Extinguished fires have zero values
                obs[4 + fire_idx*3] = 0.0
                obs[4 + fire_idx*3 + 1] = 0.0
                obs[4 + fire_idx*3 + 2] = 0.0
            fire_idx += 1

        return obs

    def step(self, action):
        """
        Execute one step in the environment
        
        Args:
            action: Integer action (0-4)
                0-3: Movement (North/South/West/East)
                4: Extinguish fire at current location
        
        Returns:
            observation: New observation
            reward: Reward for this step
            terminated: Whether episode ended (success or failure)
            truncated: Whether episode timed out
            info: Additional information
        """
        self.steps += 1
        reward = -0.1  # Time penalty
        terminated = False
        truncated = self.steps >= self.max_steps

        # Navigation actions (0-3)
        if action < 4:
            if self.water_capacity > 0:
                # Movement deltas: North(-1,0), South(+1,0), West(0,-1), East(0,+1)
                deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
                new_pos = self.drone_position + np.array(deltas, dtype=np.int32)

                # Check if movement is valid (within grid bounds)
                if 0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size:
                    self.drone_position = new_pos
                    self.water_capacity -= 1  # Water consumption for movement

                    # Reward for approaching active fires
                    active_fires = [i for i in range(self.num_fires)
                                   if not self.extinguished_fires[i]]
                    if active_fires:
                        distances = [np.linalg.norm(self.drone_position.astype(float) - 
                                   self.fire_locations[i].astype(float))
                                   for i in active_fires]
                        min_dist = np.min(distances)
                        reward += 2.0 / (1.0 + min_dist)
                else:
                    # Wall collision penalty
                    reward -= 0.2
            else:
                # Tried to move without water
                reward -= 0.1

        # Extinguish fire action (4)
        elif action == 4:
            fire_extinguished = False
            
            # Check if drone is at a fire location
            for i, fire_pos in enumerate(self.fire_locations):
                if not self.extinguished_fires[i] and \
                   np.array_equal(self.drone_position, fire_pos):
                    # Successfully extinguished fire
                    self.extinguished_fires[i] = True
                    reward += 200.0
                    fire_extinguished = True
                    # Refill water (simulates returning to base)
                    self.water_capacity = min(self.max_water_capacity, 
                                             self.water_capacity + 30)
                    break

            if not fire_extinguished:
                # Invalid extinguish attempt
                reward -= 0.5

        # Check for mission completion
        fires_extinguished = np.sum(self.extinguished_fires)
        if fires_extinguished == self.num_fires:
            # All fires extinguished - SUCCESS!
            reward += 500.0
            # Bonus for water efficiency
            reward += self.water_capacity * 0.5
            terminated = True

        # Check for failure conditions
        if self.water_capacity <= 0 and not terminated:
            # Ran out of water
            reward -= 10.0
            terminated = True

        if truncated and not terminated:
            # Episode timeout
            reward -= 5.0

        return self._get_observation(), reward, terminated, truncated, {}

    def render(self):
        """
        Render the environment
        Delegates to renderer object (set by rendering.py)
        """
        if self.render_mode is None:
            return
        
        # Import and use renderer
        if self.renderer is None:
            from environment.rendering import WildfireRenderer
            self.renderer = WildfireRenderer(self)
        
        return self.renderer.render()

    def close(self):
        """Clean up resources"""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def get_state_dict(self):
        """
        Get complete state dictionary (useful for debugging/visualization)
        
        Returns:
            dict: Complete environment state
        """
        return {
            'drone_position': self.drone_position.tolist(),
            'fire_locations': [f.tolist() for f in self.fire_locations],
            'extinguished_fires': self.extinguished_fires.tolist(),
            'water_capacity': int(self.water_capacity),
            'max_water_capacity': int(self.max_water_capacity),
            'steps': int(self.steps),
            'max_steps': int(self.max_steps),
            'num_fires': int(self.num_fires),
            'grid_size': int(self.grid_size)
        }


class WildfireCNNWrapper(gym.ObservationWrapper):
    """
    CNN Image Wrapper for Wildfire Environment
    Converts vector observations to 84x84 grayscale images for CNN processing
    
    IMPORTANT: This maintains compatibility with models trained on the
    original mission environment by using identical pixel values.
    """
    
    def __init__(self, env, img_size=84):
        super().__init__(env)
        self.img_size = img_size
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(img_size, img_size, 1), dtype=np.uint8
        )
    
    def observation(self, obs):

        img = np.zeros((self.img_size, self.img_size, 1), dtype=np.uint8)
        
        base_env = self.env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        
        drone_x = int((obs[0] / base_env.grid_size) * (self.img_size - 1))
        drone_y = int((obs[1] / base_env.grid_size) * (self.img_size - 1))
        img[max(0, drone_y-2):min(self.img_size, drone_y+3),
            max(0, drone_x-2):min(self.img_size, drone_x+3), 0] = 255
        
        # Fire locations
        for i in range(base_env.num_fires):
            fire_x = int((base_env.fire_locations[i][0] / base_env.grid_size) * (self.img_size - 1))
            fire_y = int((base_env.fire_locations[i][1] / base_env.grid_size) * (self.img_size - 1))
            
            if not base_env.extinguished_fires[i]:
                # Active fire (150 - same as training)
                img[max(0, fire_y-1):min(self.img_size, fire_y+2),
                    max(0, fire_x-1):min(self.img_size, fire_x+2), 0] = 150
            else:
                # Extinguished fire (100 - same as training)
                img[max(0, fire_y-1):min(self.img_size, fire_y+2),
                    max(0, fire_x-1):min(self.img_size, fire_x+2), 0] = 100
        
        # Water capacity bar (top of image)
        water_pct = int((obs[2] / 200) * 20)
        img[0:5, 0:water_pct, 0] = 200
        
        return img


gym.register(
    id="WildfireDrone-v0",
    entry_point="environment.custom_env:WildfireDroneEnvironment",
    max_episode_steps=500
)