"""
rendering.py
Visualization and GUI components for Wildfire Drone Environment

This file handles all Pygame rendering logic separately from environment logic.
Provides advanced visualization with animated fires, drone, and forest terrain.
"""

import pygame
import numpy as np


class WildfireRenderer:
    """
    Pygame renderer for Wildfire Drone Environment
    
    Features:
    - Realistic forest terrain with trees
    - Animated fire effects with smoke
    - Drone with rotating propellers
    - Information panel with mission status
    - Wind direction indicator
    - Real-time statistics
    """
    
    def __init__(self, env):
        """
        Initialize renderer
        
        Args:
            env: WildfireDroneEnvironment instance
        """
        self.env = env
        self.window = None
        self.clock = None
        
        # Display settings
        self.window_width = 700
        self.window_height = 750
        self.grid_display_size = 600
        
        # Initialize pygame
        pygame.init()
        
    def render(self):
        """Render the current state of the environment"""
        if self.env.render_mode is None:
            return
        
        # Create window on first render
        if self.window is None and self.env.render_mode == "human":
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption("🚁 Wildfire Drone Coordination System 🔥")
            self.clock = pygame.time.Clock()
        
        # Create canvas for grid area
        canvas = pygame.Surface((self.grid_display_size, self.grid_display_size))
        pix = self.grid_display_size / self.env.grid_size
        
        # Draw all components
        self._draw_forest_background(canvas, pix)
        self._draw_fires(canvas, pix)
        self._draw_drone(canvas, pix)
        self._draw_grid_lines(canvas, pix)
        
        # Render to window
        if self.env.render_mode == "human":
            # Process events to prevent freezing
            if pygame.get_init():
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.close()
                        return
            
            self.window.fill((40, 40, 40))  # Dark background
            self.window.blit(canvas, (0, 0))
            
            # Draw info panel
            self._draw_info_panel()
            
            pygame.display.flip()
            self.clock.tick(30)
    
    def _draw_forest_background(self, canvas, pix):
        """Draw realistic forest terrain"""
        for x in range(self.env.grid_size):
            for y in range(self.env.grid_size):
                # Checkerboard shading for depth
                shade = 10 if (x + y) % 2 == 0 else 0
                base_green = 34 + shade
                canvas.fill((base_green, 139 + shade, base_green),
                           rect=(x*pix, y*pix, pix, pix))
                
                # Draw trees (deterministic per cell)
                np.random.seed(x * 100 + y)
                num_trees = np.random.randint(2, 5)
                for _ in range(num_trees):
                    tree_x = int(x * pix + np.random.randint(5, pix-5))
                    tree_y = int(y * pix + np.random.randint(5, pix-5))
                    tree_size = np.random.randint(3, 8)
                    
                    # Tree trunk
                    pygame.draw.rect(canvas, (101, 67, 33),
                                   (tree_x - 1, tree_y, 2, tree_size))
                    # Tree foliage
                    pygame.draw.circle(canvas, (0, 100, 0),
                                     (tree_x, tree_y - tree_size//2),
                                     tree_size)
        np.random.seed()  # Reset
    
    def _draw_fires(self, canvas, pix):
        """Draw animated fires with smoke effects"""
        time_factor = self.env.steps * 0.15
        
        for i, fire_pos in enumerate(self.env.fire_locations):
            center_x = int((fire_pos[0]+0.5)*pix)
            center_y = int((fire_pos[1]+0.5)*pix)
            
            if self.env.extinguished_fires[i]:
                # Extinguished - charred area with smoke remnants
                pygame.draw.circle(canvas, (30, 30, 30),
                                 (center_x, center_y), int(pix/2))
                
                # Gray smoke
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
                
                # Checkmark
                check_size = int(pix/3)
                pygame.draw.line(canvas, (0, 255, 0),
                               (center_x - check_size//2, center_y),
                               (center_x - check_size//4, center_y + check_size//2), 4)
                pygame.draw.line(canvas, (0, 255, 0),
                               (center_x - check_size//4, center_y + check_size//2),
                               (center_x + check_size//2, center_y - check_size//2), 4)
            else:
                # Active fire - animated flames
                pulse = abs(np.sin(time_factor))
                
                # Burned ground
                pygame.draw.circle(canvas, (50, 25, 0),
                                 (center_x, center_y), int(pix/2.2))
                
                # Multi-layer flames
                for layer in range(3):
                    pulse_layer = abs(np.sin(time_factor + layer * 0.5))
                    flame_h = int(pix/2 * (1.2 + 0.3 * pulse_layer))
                    flame_w = int(pix/2.5 * (1.0 + 0.2 * pulse_layer))
                    
                    # Color gradient
                    if layer == 0:
                        color = (255, int(50 + 50 * pulse_layer), 0)
                    elif layer == 1:
                        color = (255, int(140 + 40 * pulse_layer), 0)
                    else:
                        color = (255, 255, int(100 + 100 * pulse_layer))
                    
                    flame_rect = pygame.Rect(center_x - flame_w, center_y - flame_h,
                                            flame_w * 2, flame_h * 2)
                    pygame.draw.ellipse(canvas, color, flame_rect)
                
                # Rising smoke
                for p in range(5):
                    smoke_offset = p * 15
                    smoke_x = center_x + int(10 * np.sin(time_factor + p))
                    smoke_y = center_y - 30 - smoke_offset - int(10 * pulse)
                    smoke_size = 8 + p * 3
                    smoke_alpha = 150 - p * 30
                    
                    smoke_surf = pygame.Surface((smoke_size*2, smoke_size*2))
                    smoke_surf.set_alpha(smoke_alpha)
                    smoke_surf.fill((80, 80, 80))
                    pygame.draw.circle(smoke_surf, (80, 80, 80),
                                     (smoke_size, smoke_size), smoke_size)
                    canvas.blit(smoke_surf, (smoke_x - smoke_size, smoke_y - smoke_size))
                
                # Intensity indicator
                intensity = 3 - (self.env.steps // 100) % 3
                font = pygame.font.Font(None, 20)
                intensity_text = font.render(f"⚠{intensity}", True, (255, 255, 0))
                canvas.blit(intensity_text, (center_x - 15, center_y + int(pix/2) + 5))
    
    def _draw_drone(self, canvas, pix):
        """Draw animated drone with rotating propellers"""
        center_x = int((self.env.drone_position[0]+0.5)*pix)
        center_y = int((self.env.drone_position[1]+0.5)*pix)
        
        # Shadow
        shadow_surf = pygame.Surface((int(pix/2), int(pix/2)))
        shadow_surf.set_alpha(80)
        shadow_surf.fill((0, 0, 0))
        canvas.blit(shadow_surf, (center_x - int(pix/4) + 3, 
                                  center_y - int(pix/4) + 3))
        
        # Drone arms
        arm_length = int(pix/2.5)
        pygame.draw.line(canvas, (50, 50, 50),
                       (center_x - arm_length, center_y),
                       (center_x + arm_length, center_y), 3)
        pygame.draw.line(canvas, (50, 50, 50),
                       (center_x, center_y - arm_length),
                       (center_x, center_y + arm_length), 3)
        
        # Rotating propellers
        prop_offset = int(pix/2.8)
        rotation = (self.env.steps * 0.5) % 360
        for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            prop_x = center_x + dx * prop_offset
            prop_y = center_y + dy * prop_offset
            
            # Propeller blur
            for angle_offset in [0, 45, 90, 135]:
                angle = np.radians(rotation + angle_offset)
                line_len = int(pix/6)
                end_x = int(prop_x + line_len * np.cos(angle))
                end_y = int(prop_y + line_len * np.sin(angle))
                pygame.draw.line(canvas, (200, 200, 220),
                               (prop_x, prop_y), (end_x, end_y), 2)
            
            # Hub
            pygame.draw.circle(canvas, (150, 150, 150),
                             (prop_x, prop_y), int(pix/12))
        
        # Drone body
        pygame.draw.circle(canvas, (0, 120, 255), (center_x, center_y), int(pix/4))
        pygame.draw.circle(canvas, (100, 180, 255), 
                         (center_x - 3, center_y - 3), int(pix/6))
        
        # Water tank
        if self.env.water_capacity > 0:
            tank_color = (0, 150, 255) if self.env.water_capacity > 50 else (255, 150, 0)
            pygame.draw.circle(canvas, tank_color,
                             (center_x, center_y + int(pix/5)), int(pix/8))
    
    def _draw_grid_lines(self, canvas, pix):
        """Draw forest sector boundaries"""
        for i in range(self.env.grid_size + 1):
            pygame.draw.line(canvas, (80, 80, 60), 
                           (i*pix, 0), (i*pix, self.grid_display_size), 1)
            pygame.draw.line(canvas, (80, 80, 60), 
                           (0, i*pix), (self.grid_display_size, i*pix), 1)
    
    def _draw_info_panel(self):
        """Draw information panel with mission statistics"""
        font_large = pygame.font.Font(None, 40)
        font_small = pygame.font.Font(None, 28)
        info_y = 615
        
        # Mission status
        fires_left = self.env.num_fires - np.sum(self.env.extinguished_fires)
        if fires_left == 0:
            status_text = "✅ ALL FIRES EXTINGUISHED!"
            status_color = (0, 255, 0)
        else:
            status_text = f"🔥 {fires_left} FIRES ACTIVE"
            status_color = (255, 100, 0)
        
        status = font_large.render(status_text, True, status_color)
        self.window.blit(status, (20, info_y))
        
        # Water capacity
        info_y += 45
        water_text = font_small.render("💧 Water/Retardant:", True, (255, 255, 255))
        self.window.blit(water_text, (20, info_y))
        
        # Water bar
        bar_x, bar_y = 220, info_y + 2
        bar_width, bar_height = 250, 22
        pygame.draw.rect(self.window, (60, 60, 60), 
                       (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        
        water_fill = int(bar_width * (self.env.water_capacity / self.env.max_water_capacity))
        if self.env.water_capacity > 120:
            water_color = (0, 200, 255)
        elif self.env.water_capacity > 60:
            water_color = (255, 200, 0)
        else:
            water_color = (255, 50, 0)
        
        if water_fill > 0:
            pygame.draw.rect(self.window, water_color, 
                           (bar_x, bar_y, water_fill, bar_height), border_radius=5)
        
        water_amount = font_small.render(
            f"{self.env.water_capacity}/{self.env.max_water_capacity}L", 
            True, (255, 255, 255))
        self.window.blit(water_amount, (bar_x + bar_width + 10, info_y))
        
        # Mission progress
        info_y += 35
        progress_text = font_small.render(
            f"🎯 Mission: {np.sum(self.env.extinguished_fires)}/{self.env.num_fires} fires controlled", 
            True, (200, 200, 200))
        self.window.blit(progress_text, (20, info_y))
        
        time_text = font_small.render(
            f"⏱️ Time: {self.env.steps}/{self.env.max_steps} steps", 
            True, (200, 200, 200))
        self.window.blit(time_text, (400, info_y))
        
        # Wind indicator
        info_y += 35
        wind_text = font_small.render("💨 Wind:", True, (200, 200, 200))
        self.window.blit(wind_text, (20, info_y))
        
        wind_angle = (self.env.steps * 2) % 360
        wind_x, wind_y = 130, info_y + 12
        arrow_len = 25
        wind_end_x = wind_x + int(arrow_len * np.cos(np.radians(wind_angle)))
        wind_end_y = wind_y + int(arrow_len * np.sin(np.radians(wind_angle)))
        pygame.draw.line(self.window, (100, 200, 255),
                       (wind_x, wind_y), (wind_end_x, wind_end_y), 3)
        pygame.draw.circle(self.window, (100, 200, 255),
                         (wind_end_x, wind_end_y), 5)
        
        # Legend
        legend_x = 250
        legend_text = font_small.render("Legend:", True, (150, 150, 150))
        self.window.blit(legend_text, (legend_x, info_y))
        
        pygame.draw.circle(self.window, (255, 100, 0), (legend_x + 80, info_y + 10), 8)
        legend_fire = font_small.render("Active Fire", True, (200, 200, 200))
        self.window.blit(legend_fire, (legend_x + 95, info_y))
        
        pygame.draw.circle(self.window, (30, 30, 30), (legend_x + 210, info_y + 10), 8)
        legend_ext = font_small.render("Extinguished", True, (200, 200, 200))
        self.window.blit(legend_ext, (legend_x + 225, info_y))
    
    def close(self):
        """Clean up pygame resources"""
        if self.window:
            pygame.quit()
            self.window = None