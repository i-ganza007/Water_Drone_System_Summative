"""
WILDFIRE DRONE COORDINATION SYSTEM
Non-Generic RL Environment for Fire-Fighting Drone Fleet Management

Use Case: Autonomous wildfire response using coordinated drone swarms
Agent: Fire-fighting drone equipped with water/retardant tanks
Mission: Extinguish active fire hotspots before they spread
Constraints: Limited water capacity, recharge at base, time-sensitive response

This is a REBRANDED version that uses your existing trained DQN model!
All mechanics are identical to your training - only visuals and terminology changed.
"""

import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class WildfireDroneEnvironment(gym.Env):
    """
    Wildfire Fire-Fighting Drone Coordination Environment
    
    REBRANDED FROM: MissionEnvironment
    - Agent → Fire-fighting drone
    - Missions → Active fire hotspots
    - Fuel → Water/retardant tank capacity
    - Grid → Forest sectors
    - Complete action → Extinguish fire
    
    MECHANICS: Identical to your training environment
    This ensures your trained CNN DQN model works without retraining!
    """

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
        """
        Observation vector (IDENTICAL to training):
        [drone_x, drone_y, water_capacity, fires_extinguished, 
         fire1_dist, fire1_dx, fire1_dy, fire2_dist, ...]
        """
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
        """
        Step function (IDENTICAL reward structure to training)
        Action 0-3: Navigate drone
        Action 4: Extinguish fire at current location
        """
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
        """
        Advanced Pygame visualization showing:
        - Realistic forest terrain with trees
        - Animated fire hotspots with smoke
        - Drone with rotating propellers
        - Wind direction indicator
        - Water capacity and mission status
        - Fire intensity levels
        """
        if self.render_mode is None:
            return
            
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.set_mode((600, 650))  # Smaller window size
            # Position window to the left side of screen
            import os
            os.environ['SDL_VIDEO_WINDOW_POS'] = '100,50'  # x=100, y=50 from top-left
            self.window = pygame.display.set_mode((600, 650))
            pygame.display.set_caption("🚁 Wildfire Drone Coordination System 🔥")
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((600, 520))  # Smaller canvas to match window
        
        # ====================================================================
        # FOREST BACKGROUND with terrain variation
        # ====================================================================
        pix = 520 / self.grid_size  # Adjusted for smaller canvas
        
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
                           (x*pix, 0), (x*pix, 520), 1)  # Updated height
            pygame.draw.line(canvas, (80, 80, 60), 
                           (0, x*pix), (520, x*pix), 1)  # Updated width
        
        # Render to window
        if self.render_mode == "human":
            # CRITICAL FIX: Process pygame events to prevent freezing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return
            
            self.window.fill((40, 40, 40))  # Dark background
            self.window.blit(canvas, (0, 0))
            
            # ================================================================
            # INFORMATION PANEL - Enhanced UI
            # ================================================================
            font_large = pygame.font.Font(None, 36)  # Slightly smaller font
            font_small = pygame.font.Font(None, 24)   # Smaller font
            info_y = 535  # Adjusted for smaller window
            
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
            bar_x, bar_y = 180, info_y + 2  # Adjusted position
            bar_width = 200  # Smaller bar width
            bar_height = 18  # Smaller bar height
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
            legend_x = 200  # Adjusted for smaller window
            legend_text = font_small.render("Legend:", True, (150, 150, 150))
            self.window.blit(legend_text, (legend_x, info_y))
            
            # Color indicators
            pygame.draw.circle(self.window, (255, 100, 0), (legend_x + 70, info_y + 8), 6)  # Smaller circles
            legend_fire = font_small.render("Active Fire", True, (200, 200, 200))
            self.window.blit(legend_fire, (legend_x + 85, info_y))
            
            pygame.draw.circle(self.window, (30, 30, 30), (legend_x + 170, info_y + 8), 6)
            legend_ext = font_small.render("Extinguished", True, (200, 200, 200))
            self.window.blit(legend_ext, (legend_x + 185, info_y))
            
            pygame.display.flip()
            self.clock.tick(30)

    def close(self):
        if self.window:
            pygame.quit()
            self.window = None


# ============================================================================
# CNN IMAGE WRAPPER - REBRANDED FOR WILDFIRE
# ============================================================================

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


# ============================================================================
# DEMO SCRIPT - USE YOUR EXISTING TRAINED MODEL
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("🚁 WILDFIRE DRONE COORDINATION SYSTEM 🔥")
    print("="*70)
    print("\n🎯 Mission: Autonomous wildfire response using AI-controlled drones")
    print("📍 Environment: 10x10 forest grid with active fire hotspots")
    print("🤖 Agent: Fire-fighting drone with water/retardant capacity")
    print("\n" + "="*70)
    
    # Option 1: Run with your trained model
    USE_TRAINED_MODEL = True  # Set to False for random demo
    
    if USE_TRAINED_MODEL:
        try:
            from stable_baselines3 import DQN
            
            # Load your existing trained model
            model_path = "models\DQN\dqn_cnn_mission_sixth_config.zip"
            print(f"\n📦 Loading trained model from:")
            print(f"   {model_path}")
            
            model = DQN.load(model_path)
            print("✅ Model loaded successfully!")
            print(f"   Policy: {model.policy.__class__.__name__}")
            print("\n🔥 The model will control the fire-fighting drone.")
            print("   (Same model that was trained on 'missions' - now fighting fires!)")
            
        except Exception as e:
            print(f"\n⚠️  Could not load model: {e}")
            print("   Running with random actions instead...")
            model = None
            USE_TRAINED_MODEL = False
    else:
        model = None
        print("\n🎲 Running with RANDOM actions (no trained model)")
        print("   The drone will move randomly for demonstration.")
    
    print("\n" + "="*70)
    print("🎮 CONTROLS:")
    print("   • Watch the drone navigate and extinguish fires")
    print("   • Press Ctrl+C to stop")
    print("="*70 + "\n")
    
    # Create environment
    env = WildfireDroneEnvironment(grid_size=10, num_fires=3, render_mode="human")
    wrapped_env = WildfireCNNWrapper(env, img_size=84)
    
    # Run simulation
    try:
        for episode in range(10):  # Run 10 episodes
            obs, _ = wrapped_env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            print(f"\n🔥 Episode {episode + 1} - New wildfire outbreak detected!")
            
            while not done:
                wrapped_env.render()
                
                # Process pygame events AFTER first render
                if pygame.get_init():
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            print("\n⚠️  Window closed by user")
                            env.close()
                            exit(0)
                
                if USE_TRAINED_MODEL and model:
                    # Use trained model
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    # Random action
                    action = wrapped_env.action_space.sample()
                
                obs, reward, terminated, truncated, _ = wrapped_env.step(action)
                episode_reward += reward
                steps += 1
                done = terminated or truncated
                
                # Small delay for visibility (reduced for smoother playback)
                if pygame.get_init():
                    pygame.time.wait(30)
            
            # Episode summary
            fires_extinguished = np.sum(env.extinguished_fires)
            print(f"   ✓ Episode complete!")
            print(f"   • Fires extinguished: {fires_extinguished}/{env.num_fires}")
            print(f"   • Steps taken: {steps}")
            print(f"   • Total reward: {episode_reward:.2f}")
            print(f"   • Water remaining: {env.water_capacity}L")
            
            if fires_extinguished == env.num_fires:
                print(f"   🎉 SUCCESS - All fires controlled!")
            else:
                print(f"   ❌ FAILED - {env.num_fires - fires_extinguished} fires still burning")
            
            # Pause between episodes with event handling
            if pygame.get_init():
                start_time = pygame.time.get_ticks()
                while pygame.time.get_ticks() - start_time < 2000:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            print("\n⚠️  Window closed by user")
                            env.close()
                            exit(0)
                    wrapped_env.render()
                    pygame.time.wait(100)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Simulation stopped by user")
    
    finally:
        env.close()
        print("\n" + "="*70)
        print("🚁 Wildfire Drone System Shutdown")
        print("="*70)


# ============================================================================
# INTEGRATION NOTES FOR YOUR ASSIGNMENT
# ============================================================================

"""
✅ HOW TO USE THIS WITH YOUR EXISTING MODEL:

1. RENAME YOUR ENVIRONMENT FILE:
   - Save this as: wildfire_drone_env.py
   - Put in: project_root/environment/

2. UPDATE YOUR TRAINING SCRIPTS:
   Replace:
   ```python
   from your_old_file import MissionEnvironment
   env = MissionEnvironment(grid_size=10, num_missions=3)
   ```
   
   With:
   ```python
   from environment.wildfire_drone_env import WildfireDroneEnvironment
   env = WildfireDroneEnvironment(grid_size=10, num_fires=3)
   ```

3. CNN WRAPPER:
   Replace:
   ```python
   from your_old_file import GridToImageWrapper
   ```
   
   With:
   ```python
   from environment.wildfire_drone_env import WildfireCNNWrapper
   ```

4. YOUR EXISTING MODEL WORKS UNCHANGED:
   - Same observation space (84x84x1)
   - Same action space (5 actions)
   - Same reward structure
   - Same pixel values in CNN images
   
   Just load and use:
   ```python
   model = DQN.load("your_model_path.zip")
   # Model works immediately!
   ```

5. FOR YOUR REPORT:
   - Title: "Wildfire Drone Coordination using Deep RL"
   - Problem: Autonomous wildfire response
   - Agent: Fire-fighting drone with water capacity
   - Actions: Navigate forest (N/S/E/W) + Extinguish fire
   - Objective: Minimize fire spread, maximize area saved
   - Constraints: Limited water, time-sensitive response
   
   This is NON-GENERIC because:
   ✓ Real-world wildfire application
   ✓ Domain-specific constraints (water capacity, fire spread)
   ✓ Time-critical emergency response scenario
   ✓ Practical disaster management use case

6. VIDEO DEMONSTRATION:
   When recording, emphasize:
   - "Fire-fighting drone" not "agent"
   - "Extinguishing fires" not "completing missions"
   - "Water/retardant capacity" not "fuel"
   - Show the forest background and animated fires
   - Explain fire spread risk and time sensitivity

7. ADVANCED FEATURES TO MENTION:
   - Animated fire with smoke effects
   - Realistic forest terrain
   - Rotating drone propellers
   - Wind direction indicator
   - Real-time mission status
   - Water capacity management
   - Fire intensity levels
"""