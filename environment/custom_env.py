import pygame
import numpy as np
import time
import sys
import os
import gymnasium as gym
from gymnasium import spaces

# Move the environment classes here to avoid circular imports
class WildfireDroneEnvironment(gym.Env):


    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, grid_size=10, num_fires=3, render_mode=None, 
                 enable_fire_spread=False):
        super().__init__()
        self.grid_size = grid_size
        self.num_fires = num_fires  # Number of active fire hotspots
        self.render_mode = render_mode
        self.enable_fire_spread = enable_fire_spread  # Optional: disable for model compatibility

        # Action space: 0=North, 1=South, 2=West, 3=East, 4=Extinguish
        self.action_space = spaces.Discrete(5)

        # Observation space (same as training)
        obs_size = 4 + (num_fires * 3)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0] + [0, -grid_size, -grid_size] * num_fires, dtype=np.float32),
            high=np.array([grid_size, grid_size, 200, num_fires] + 
                         [np.sqrt(2*grid_size**2), grid_size, grid_size] * num_fires, dtype=np.float32),
            dtype=np.float32
        )

        # Drone state
        self.drone_position = None
        self.fire_locations = None  # Active fire hotspots
        self.extinguished_fires = None
        self.water_capacity = 200  # Water/retardant tank capacity
        self.max_water_capacity = 200
        self.steps = 0
        self.max_steps = 50
        
        # Pygame rendering
        self.window = None
        self.clock = None
        self.fire_sprites = {}
        self.drone_sprite = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize drone at random starting position
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

        obs = np.zeros(4 + self.num_fires * 3, dtype=np.float32)
        obs[0] = float(self.drone_position[0])
        obs[1] = float(self.drone_position[1])
        obs[2] = float(self.water_capacity)
        obs[3] = float(np.sum(self.extinguished_fires))

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
                obs[4 + fire_idx*3] = 0.0
                obs[4 + fire_idx*3 + 1] = 0.0
                obs[4 + fire_idx*3 + 2] = 0.0
            fire_idx += 1

        return obs

    def step(self, action):

        self.steps += 1
        reward = -0.1  # Time penalty (encourage efficiency)
        terminated = False
        truncated = self.steps >= self.max_steps

        # Navigation actions (0-3)
        if action < 4:
            if self.water_capacity > 0:
                # Movement deltas: North, South, West, East
                deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
                new_pos = self.drone_position + np.array(deltas, dtype=np.int32)

                # Valid movement within forest sectors
                if 0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size:
                    self.drone_position = new_pos
                    self.water_capacity -= 1  # Water usage for propulsion

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
                    # Collision with forest boundary
                    reward -= 0.2
            else:
                # Out of water
                reward -= 0.1

        # Extinguish action (4)
        elif action == 4:
            fire_extinguished = False
            
            for i, fire_pos in enumerate(self.fire_locations):
                if not self.extinguished_fires[i] and \
                   np.array_equal(self.drone_position, fire_pos):
                    # Successfully extinguished fire
                    self.extinguished_fires[i] = True
                    reward += 200.0  # Major reward for saving area
                    fire_extinguished = True
                    # Refill water at base (simulated)
                    self.water_capacity = min(self.max_water_capacity, 
                                             self.water_capacity + 30)
                    break

            if not fire_extinguished:
                # Attempted extinguish without fire present
                reward -= 0.5

        # Check mission completion
        fires_extinguished = np.sum(self.extinguished_fires)
        if fires_extinguished == self.num_fires:
            # All fires extinguished - mission success!
            reward += 500.0
            # Bonus for remaining water (efficiency)
            reward += self.water_capacity * 0.5
            terminated = True

        # Failure conditions
        if self.water_capacity <= 0 and not terminated:
            # Ran out of water before completing mission
            reward -= 10.0
            terminated = True

        if truncated and not terminated:
            # Timeout - fires spread uncontrolled
            reward -= 5.0

        return self._get_observation(), reward, terminated, truncated, {}

    def render(self):

        if self.render_mode is None:
            return
            
        if self.window is None and self.render_mode == "human":
            pygame.init()
            self.window = pygame.display.set_mode((700, 750))  # Wider for legend
            pygame.display.set_caption("🚁 Wildfire Drone Coordination System 🔥")
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((700, 600))
        
        # ====================================================================
        # FOREST BACKGROUND with terrain variation
        # ====================================================================
        pix = 600 / self.grid_size
        
        # Create varied forest terrain (different shades of green)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                # Checkerboard pattern for depth
                shade_variation = 10 if (x + y) % 2 == 0 else 0
                base_green = 34 + shade_variation
                canvas.fill((base_green, 139 + shade_variation, base_green),
                           rect=(x*pix, y*pix, pix, pix))
                
                # Draw random trees in each cell (deterministic based on position)
                np.random.seed(x * 100 + y)
                num_trees = np.random.randint(2, 5)
                for _ in range(num_trees):
                    tree_x = int(x * pix + np.random.randint(5, pix-5))
                    tree_y = int(y * pix + np.random.randint(5, pix-5))
                    tree_size = np.random.randint(3, 8)
                    
                    # Tree trunk (brown)
                    pygame.draw.rect(canvas, (101, 67, 33),
                                   (tree_x - 1, tree_y, 2, tree_size))
                    # Tree foliage (dark green circle)
                    pygame.draw.circle(canvas, (0, 100, 0),
                                     (tree_x, tree_y - tree_size//2),
                                     tree_size)
        
        np.random.seed()  # Reset seed
        
        # ====================================================================
        # FIRE HOTSPOTS with animated flames and smoke
        # ====================================================================
        for i, fire_pos in enumerate(self.fire_locations):
            center_x = int((fire_pos[0]+0.5)*pix)
            center_y = int((fire_pos[1]+0.5)*pix)
            
            if self.extinguished_fires[i]:
                # EXTINGUISHED - Show charred/burned area
                # Blackened ground
                pygame.draw.circle(canvas, (30, 30, 30),
                                 (center_x, center_y),
                                 int(pix/2))
                # Gray smoke remnants
                for j in range(3):
                    smoke_y = center_y - 10 - j*8
                    smoke_alpha = 100 - j*30
                    smoke_size = int(pix/4) + j*2
                    smoke_surf = pygame.Surface((smoke_size*2, smoke_size*2))
                    smoke_surf.set_alpha(smoke_alpha)
                    smoke_surf.fill((150, 150, 150))
                    pygame.draw.circle(smoke_surf, (150, 150, 150),
                                     (smoke_size, smoke_size), smoke_size)
                    canvas.blit(smoke_surf, 
                              (center_x - smoke_size, smoke_y - smoke_size))
                
                # Green checkmark overlay
                check_size = int(pix/3)
                pygame.draw.line(canvas, (0, 255, 0),
                               (center_x - check_size//2, center_y),
                               (center_x - check_size//4, center_y + check_size//2),
                               4)
                pygame.draw.line(canvas, (0, 255, 0),
                               (center_x - check_size//4, center_y + check_size//2),
                               (center_x + check_size//2, center_y - check_size//2),
                               4)
            else:
                # ACTIVE FIRE - Animated flames
                time_factor = self.steps * 0.15
                
                # Burned ground underneath
                pygame.draw.circle(canvas, (50, 25, 0),
                                 (center_x, center_y),
                                 int(pix/2.2))
                
                # Multiple flame layers for depth
                for layer in range(3):
                    pulse = abs(np.sin(time_factor + layer * 0.5))
                    flame_height = int(pix/2 * (1.2 + 0.3 * pulse))
                    flame_width = int(pix/2.5 * (1.0 + 0.2 * pulse))
                    
                    # Color gradient (red -> orange -> yellow)
                    if layer == 0:  # Outer flame (red)
                        color = (255, int(50 + 50 * pulse), 0)
                    elif layer == 1:  # Middle flame (orange)
                        color = (255, int(140 + 40 * pulse), 0)
                    else:  # Inner flame (yellow)
                        color = (255, 255, int(100 + 100 * pulse))
                    
                    # Draw flame shape (ellipse)
                    flame_rect = pygame.Rect(
                        center_x - flame_width,
                        center_y - flame_height,
                        flame_width * 2,
                        flame_height * 2
                    )
                    pygame.draw.ellipse(canvas, color, flame_rect)
                
                # SMOKE - Rising particles
                for smoke_particle in range(5):
                    smoke_offset = smoke_particle * 15
                    smoke_x = center_x + int(10 * np.sin(time_factor + smoke_particle))
                    smoke_y = center_y - 30 - smoke_offset - int(10 * pulse)
                    smoke_size = 8 + smoke_particle * 3
                    smoke_alpha = 150 - smoke_particle * 30
                    
                    smoke_surf = pygame.Surface((smoke_size*2, smoke_size*2))
                    smoke_surf.set_alpha(smoke_alpha)
                    smoke_surf.fill((80, 80, 80))
                    pygame.draw.circle(smoke_surf, (80, 80, 80),
                                     (smoke_size, smoke_size), smoke_size)
                    canvas.blit(smoke_surf, 
                              (smoke_x - smoke_size, smoke_y - smoke_size))
                
                # Fire intensity indicator (small flame icon with number)
                intensity = 3 - (self.steps // 100) % 3  # Varies over time
                intensity_text = pygame.font.Font(None, 20).render(
                    f"⚠{intensity}", True, (255, 255, 0))
                canvas.blit(intensity_text, (center_x - 15, center_y + int(pix/2) + 5))

        # ====================================================================
        # DRONE - Animated quadcopter with rotating propellers
        # ====================================================================
        drone_center_x = int((self.drone_position[0]+0.5)*pix)
        drone_center_y = int((self.drone_position[1]+0.5)*pix)
        
        # Drone shadow (for depth)
        shadow_surf = pygame.Surface((int(pix/2), int(pix/2)))
        shadow_surf.set_alpha(80)
        shadow_surf.fill((0, 0, 0))
        canvas.blit(shadow_surf, (drone_center_x - int(pix/4) + 3, 
                                  drone_center_y - int(pix/4) + 3))
        
        # Drone arms (crossing lines)
        arm_length = int(pix/2.5)
        pygame.draw.line(canvas, (50, 50, 50),
                       (drone_center_x - arm_length, drone_center_y),
                       (drone_center_x + arm_length, drone_center_y), 3)
        pygame.draw.line(canvas, (50, 50, 50),
                       (drone_center_x, drone_center_y - arm_length),
                       (drone_center_x, drone_center_y + arm_length), 3)
        
        # Rotating propellers (4 corners)
        prop_offset = int(pix/2.8)
        rotation = (self.steps * 0.5) % 360
        for corner_idx, (dx, dy) in enumerate([(-1, -1), (1, -1), (-1, 1), (1, 1)]):
            prop_x = drone_center_x + dx * prop_offset
            prop_y = drone_center_y + dy * prop_offset
            
            # Propeller blur effect (rotating lines)
            for angle_offset in [0, 45, 90, 135]:
                angle = np.radians(rotation + angle_offset)
                line_length = int(pix/6)
                end_x = int(prop_x + line_length * np.cos(angle))
                end_y = int(prop_y + line_length * np.sin(angle))
                pygame.draw.line(canvas, (200, 200, 220),
                               (prop_x, prop_y), (end_x, end_y), 2)
            
            # Propeller hub
            pygame.draw.circle(canvas, (150, 150, 150),
                             (prop_x, prop_y), int(pix/12))
        
        # Drone body (central sphere)
        pygame.draw.circle(canvas, (0, 120, 255), 
                         (drone_center_x, drone_center_y), 
                         int(pix/4))
        # Highlight for 3D effect
        pygame.draw.circle(canvas, (100, 180, 255), 
                         (drone_center_x - 3, drone_center_y - 3), 
                         int(pix/6))
        
        # Water tank indicator on drone
        if self.water_capacity > 0:
            tank_color = (0, 150, 255) if self.water_capacity > 50 else (255, 150, 0)
            pygame.draw.circle(canvas, tank_color,
                             (drone_center_x, drone_center_y + int(pix/5)),
                             int(pix/8))

        # ====================================================================
        # GRID LINES (Forest sector boundaries)
        # ====================================================================
        for x in range(self.grid_size + 1):
            pygame.draw.line(canvas, (80, 80, 60), 
                           (x*pix, 0), (x*pix, 600), 1)
            pygame.draw.line(canvas, (80, 80, 60), 
                           (0, x*pix), (600, x*pix), 1)
        
        # Render to window
        if self.render_mode == "human":
            # Process pygame events to prevent freezing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return
            
            self.window.fill((40, 40, 40))  # Dark background
            self.window.blit(canvas, (0, 0))
            
            # ================================================================
            # INFORMATION PANEL - Enhanced UI
            # ================================================================
            font_large = pygame.font.Font(None, 40)
            font_small = pygame.font.Font(None, 28)
            info_y = 615
            
            # Mission status banner
            fires_left = self.num_fires - np.sum(self.extinguished_fires)
            if fires_left == 0:
                status_text = "✅ ALL FIRES EXTINGUISHED!"
                status_color = (0, 255, 0)
            else:
                status_text = f"🔥 {fires_left} FIRES ACTIVE"
                status_color = (255, 100, 0)
            
            status = font_large.render(status_text, True, status_color)
            self.window.blit(status, (20, info_y))
            
            # Water capacity with icon
            info_y += 45
            water_text = font_small.render(f"💧 Water/Retardant:", True, (255, 255, 255))
            self.window.blit(water_text, (20, info_y))
            
            # Progress bar for water
            bar_x, bar_y = 220, info_y + 2
            bar_width = 250
            bar_height = 22
            pygame.draw.rect(self.window, (60, 60, 60), 
                           (bar_x, bar_y, bar_width, bar_height), border_radius=5)
            water_fill = int(bar_width * (self.water_capacity / self.max_water_capacity))
            
            # Color based on capacity
            if self.water_capacity > 120:
                water_color = (0, 200, 255)
            elif self.water_capacity > 60:
                water_color = (255, 200, 0)
            else:
                water_color = (255, 50, 0)
            
            if water_fill > 0:
                pygame.draw.rect(self.window, water_color, 
                               (bar_x, bar_y, water_fill, bar_height), border_radius=5)
            
            # Water amount text
            water_amount = font_small.render(f"{self.water_capacity}/{self.max_water_capacity}L", 
                                            True, (255, 255, 255))
            self.window.blit(water_amount, (bar_x + bar_width + 10, info_y))
            
            # Mission progress
            info_y += 35
            progress_text = font_small.render(
                f"🎯 Mission: {np.sum(self.extinguished_fires)}/{self.num_fires} fires controlled", 
                True, (200, 200, 200))
            self.window.blit(progress_text, (20, info_y))
            
            # Steps/Time indicator
            time_text = font_small.render(f"⏱️ Time: {self.steps}/{self.max_steps} steps", 
                                         True, (200, 200, 200))
            self.window.blit(time_text, (400, info_y))
            
            # Wind direction indicator (visual only, doesn't affect mechanics)
            info_y += 35
            wind_text = font_small.render("💨 Wind:", True, (200, 200, 200))
            self.window.blit(wind_text, (20, info_y))
            
            # Animated wind arrow
            wind_angle = (self.steps * 2) % 360  # Rotating wind
            wind_x, wind_y = 130, info_y + 12
            arrow_length = 25
            wind_end_x = wind_x + int(arrow_length * np.cos(np.radians(wind_angle)))
            wind_end_y = wind_y + int(arrow_length * np.sin(np.radians(wind_angle)))
            pygame.draw.line(self.window, (100, 200, 255),
                           (wind_x, wind_y), (wind_end_x, wind_end_y), 3)
            pygame.draw.circle(self.window, (100, 200, 255),
                             (wind_end_x, wind_end_y), 5)
            
            # Legend
            legend_x = 250
            legend_text = font_small.render("Legend:", True, (150, 150, 150))
            self.window.blit(legend_text, (legend_x, info_y))
            
            # Color indicators
            pygame.draw.circle(self.window, (255, 100, 0), (legend_x + 80, info_y + 10), 8)
            legend_fire = font_small.render("Active Fire", True, (200, 200, 200))
            self.window.blit(legend_fire, (legend_x + 95, info_y))
            
            pygame.draw.circle(self.window, (30, 30, 30), (legend_x + 210, info_y + 10), 8)
            legend_ext = font_small.render("Extinguished", True, (200, 200, 200))
            self.window.blit(legend_ext, (legend_x + 225, info_y))
            
            pygame.display.flip()
            self.clock.tick(30)

    def close(self):
        if self.window:
            pygame.quit()
            self.window = None


class WildfireCNNWrapper(gym.ObservationWrapper):
    """
    CNN Image Wrapper for Wildfire Environment
    IDENTICAL to your training wrapper - just rebranded visuals
    
    Your trained CNN DQN model will work without any changes!
    """
    def __init__(self, env, img_size=84):
        super().__init__(env)
        self.img_size = img_size
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(img_size, img_size, 1), dtype=np.uint8
        )
    
    def observation(self, obs):
        """Convert vector observation to image (SAME as training)"""
        img = np.zeros((self.img_size, self.img_size, 1), dtype=np.uint8)
        
        base_env = self.env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        
        # Drone position (bright white - same as training)
        drone_x = int((obs[0] / base_env.grid_size) * (self.img_size - 1))
        drone_y = int((obs[1] / base_env.grid_size) * (self.img_size - 1))
        img[max(0, drone_y-2):min(self.img_size, drone_y+3),
            max(0, drone_x-2):min(self.img_size, drone_x+3), 0] = 255
        
        # Fire locations (same pixel values as training)
        for i in range(base_env.num_fires):
            fire_x = int((base_env.fire_locations[i][0] / base_env.grid_size) * (self.img_size - 1))
            fire_y = int((base_env.fire_locations[i][1] / base_env.grid_size) * (self.img_size - 1))
            
            if not base_env.extinguished_fires[i]:
                # Active fire (same as uncompleted mission in training)
                img[max(0, fire_y-1):min(self.img_size, fire_y+2),
                    max(0, fire_x-1):min(self.img_size, fire_x+2), 0] = 150
            else:
                # Extinguished (same as completed mission in training)
                img[max(0, fire_y-1):min(self.img_size, fire_y+2),
                    max(0, fire_x-1):min(self.img_size, fire_x+2), 0] = 100
        
        # Water capacity bar (same as fuel bar in training)
        water_pct = int((obs[2] / 200) * 20)
        img[0:5, 0:water_pct, 0] = 200
        
        return img


def demo_random_agent(num_episodes=5, steps_per_episode=100):

    
    print("="*70)
    print(" WILDFIRE DRONE COORDINATION SYSTEM - RANDOM AGENT DEMO 🔥")
    print("="*70)
    print("\nPurpose: Demonstrate visualization WITHOUT trained model")
    print("Agent Behavior: Takes RANDOM actions to showcase environment\n")
    print("="*70)
    
    # Create environment with rendering enabled
    env = WildfireDroneEnvironment(
        grid_size=10, 
        num_fires=3, 
        render_mode="human"
    )
    
    # Optional: Wrap with CNN wrapper to show image observations
    # wrapped_env = WildfireCNNWrapper(env, img_size=84)
    
    print("\n Environment Details:")
    print(f"  • Grid Size: {env.grid_size}x{env.grid_size}")
    print(f"  • Number of Fires: {env.num_fires}")
    print(f"  • Action Space: {env.action_space}")
    print(f"    - 0: Move North")
    print(f"    - 1: Move South")
    print(f"    - 2: Move West")
    print(f"    - 3: Move East")
    print(f"    - 4: Extinguish Fire")
    print(f"  • Observation Space: {env.observation_space.shape}")
    print(f"  • Max Water Capacity: {env.max_water_capacity}L")
    print(f"  • Max Steps: {env.max_steps}")
    
    print("\n" + "="*70)
    print(" CONTROLS:")
    print("  • Close window or press Ctrl+C to stop")
    print("  • Agent is taking RANDOM actions automatically")
    print("="*70 + "\n")
    
    # Statistics tracking
    total_fires_extinguished = 0
    total_steps = 0
    successful_episodes = 0
    
    try:
        for episode in range(1, num_episodes + 1):
            print(f"\n{'='*70}")
            print(f" EPISODE {episode}/{num_episodes} - New Wildfire Outbreak!")
            print(f"{'='*70}")
            
            # Reset environment
            obs, info = env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False
            
            # Print initial state
            print(f"\n Initial State:")
            print(f"  • Drone Position: {env.drone_position}")
            print(f"  • Fire Locations: {[f.tolist() for f in env.fire_locations]}")
            print(f"  • Water Capacity: {env.water_capacity}L\n")
            
            # Action statistics for this episode
            action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
            action_names = {
                0: "North", 1: "South", 2: "West", 3: "East", 4: "Extinguish"
            }
            
            # Run episode
            while not done and episode_steps < steps_per_episode:
                # Render current state
                env.render()
                
                # Process pygame events to prevent freezing
                if pygame.get_init():
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            print("\n  Window closed by user")
                            env.close()
                            return
                
                # RANDOM ACTION SELECTION (No trained model)
                action = env.action_space.sample()
                action_counts[action] += 1
                
                # Print action every 10 steps
                if episode_steps % 10 == 0:
                    print(f"  Step {episode_steps}: Random Action = {action} ({action_names[action]}) | "
                          f"Drone at {env.drone_position} | Water: {env.water_capacity}L | "
                          f"Fires: {np.sum(env.extinguished_fires)}/{env.num_fires}")
                
                # Execute action
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_steps += 1
                total_steps += 1
                done = terminated or truncated
                
                # Small delay for visibility
                if pygame.get_init():
                    pygame.time.wait(50)  # 50ms delay
            
            # Episode summary
            fires_extinguished = np.sum(env.extinguished_fires)
            total_fires_extinguished += fires_extinguished
            
            if fires_extinguished == env.num_fires:
                successful_episodes += 1
                result = " SUCCESS"
                result_color = "\033[92m"  # Green
            else:
                result = " FAILED"
                result_color = "\033[91m"  # Red
            
            print(f"\n{'='*70}")
            print(f"{result_color}{result}\033[0m - Episode {episode} Complete")
            print(f"{'='*70}")
            print(f" Episode Statistics:")
            print(f"  • Total Steps: {episode_steps}")
            print(f"  • Fires Extinguished: {fires_extinguished}/{env.num_fires}")
            print(f"  • Final Water: {env.water_capacity}L")
            print(f"  • Total Reward: {episode_reward:.2f}")
            print(f"\n Action Distribution (Random):")
            for action_id, count in action_counts.items():
                percentage = (count / episode_steps * 100) if episode_steps > 0 else 0
                print(f"  • {action_names[action_id]:12s}: {count:3d} times ({percentage:.1f}%)")
            
            # Reason for episode end
            if terminated:
                if fires_extinguished == env.num_fires:
                    print(f"\n Termination: All fires successfully extinguished!")
                elif env.water_capacity <= 0:
                    print(f"\n Termination: Ran out of water")
            elif truncated:
                print(f"\n⏱  Truncation: Episode timeout ({env.max_steps} steps)")
            
            print(f"{'='*70}\n")
            
            # Pause between episodes
            if pygame.get_init() and episode < num_episodes:
                print(" Next episode starting in 2 seconds...\n")
                start_time = pygame.time.get_ticks()
                while pygame.time.get_ticks() - start_time < 2000:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            print("\n  Window closed by user")
                            env.close()
                            return
                    env.render()
                    pygame.time.wait(100)
    
    except KeyboardInterrupt:
        print("\n\n  Demo interrupted by user (Ctrl+C)")
    
    finally:
        # Final summary
        print("\n" + "="*70)
        print(" OVERALL STATISTICS (Random Agent)")
        print("="*70)
        print(f"  • Total Episodes: {num_episodes}")
        print(f"  • Successful Episodes: {successful_episodes}/{num_episodes} "
              f"({successful_episodes/num_episodes*100:.1f}%)")
        print(f"  • Total Steps: {total_steps}")
        print(f"  • Average Steps per Episode: {total_steps/num_episodes:.1f}")
        print(f"  • Total Fires Extinguished: {total_fires_extinguished}/{num_episodes * env.num_fires} "
              f"({total_fires_extinguished/(num_episodes * env.num_fires)*100:.1f}%)")

        print(f"\n INSIGHTS:")
        print(f"  • Random agent success rate: {successful_episodes/num_episodes*100:.1f}%")
        print(f"  • This demonstrates the BASELINE performance")
        print(f"  • A trained RL model should perform MUCH better")
        print(f"  • Your trained DQN achieves ~60-90% success rate")
        
        print("\n" + "="*70)
        print(" Demo Complete - All Visualization Components Demonstrated")
        print("="*70)
        print("\n Next Steps:")
        print("  1. Verify all visual components are working correctly")
        print("  2. Proceed with training RL models (DQN, PPO, A2C, REINFORCE)")
        print("  3. Compare trained model performance against this baseline\n")
        
        # Clean up
        env.close()


def demo_with_user_control():
    """
    Alternative demo: User controls the drone with keyboard
    Useful for understanding game mechanics
    """
    print("="*70)
    print(" MANUAL CONTROL DEMO - YOU Control the Drone")
    print("="*70)
    print("\n  KEYBOARD CONTROLS:")
    print("  • ↑ Arrow Key: Move North")
    print("  • ↓ Arrow Key: Move South")
    print("  • ← Arrow Key: Move West")
    print("  • → Arrow Key: Move East")
    print("  • SPACE: Extinguish Fire")
    print("  • ESC: Quit")
    print("\n" + "="*70 + "\n")
    
    env = WildfireDroneEnvironment(grid_size=10, num_fires=3, render_mode="human")
    obs, _ = env.reset()
    done = False
    episode_reward = 0
    
    try:
        while not done:
            env.render()
            
            action = None
            waiting_for_input = True
            
            while waiting_for_input:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return
                    
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_UP:
                            action = 0  # North
                            waiting_for_input = False
                        elif event.key == pygame.K_DOWN:
                            action = 1  # South
                            waiting_for_input = False
                        elif event.key == pygame.K_LEFT:
                            action = 2  # West
                            waiting_for_input = False
                        elif event.key == pygame.K_RIGHT:
                            action = 3  # East
                            waiting_for_input = False
                        elif event.key == pygame.K_SPACE:
                            action = 4  # Extinguish
                            waiting_for_input = False
                        elif event.key == pygame.K_ESCAPE:
                            env.close()
                            return
                
                pygame.time.wait(10)
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
            action_names = ["North", "South", "West", "East", "Extinguish"]
            print(f"Action: {action_names[action]} | Reward: {reward:.2f} | "
                  f"Total: {episode_reward:.2f} | Fires: {np.sum(env.extinguished_fires)}/{env.num_fires}")
        
        print(f"\n{'='*70}")
        if np.sum(env.extinguished_fires) == env.num_fires:
            print(" SUCCESS! You extinguished all fires!")
        else:
            print(" Mission Failed")
        print(f"Final Reward: {episode_reward:.2f}")
        print(f"{'='*70}\n")
    
    except KeyboardInterrupt:
        print("\n Demo stopped by user")
    
    finally:
        env.close()


if __name__ == "__main__":
    import sys
    
    print("\n🚁 Wildfire Drone Coordination System - Visualization Demo\n")
    print("Choose demo mode:")
    print("  1. Random Agent (automatic)")
    print("  2. Manual Control (keyboard)\n")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        num_episodes = 3 
        demo_random_agent(num_episodes=num_episodes, steps_per_episode=200)
    
    elif choice == "2":
        demo_with_user_control()
    
    else:
        print("Invalid choice. Running default random agent demo...")
        demo_random_agent(num_episodes=3, steps_per_episode=200)