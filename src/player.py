import pygame

class Player:
    def __init__(self, name, start_pos, start_dir, color=(255, 255, 255), sprite=None): # Add sprite parameter
        self.name = name
        self.start_pos = list(start_pos) # Store start position
        self.position = list(start_pos) # Current position
        self.direction = start_dir
        self.color = color # Assign the color
        self.sprite = sprite # Store the sprite
        self.trail = [list(start_pos)] # Keep track of the trail

    def reset(self):
        self.position = list(self.start_pos)
        self.trail = [list(self.start_pos)]
        # Optionally reset direction if needed, depends on game rules
        # self.direction = self.start_dir

    def move(self):
        x, y = self.position
        if self.direction == "UP":
            y -= 1
        elif self.direction == "DOWN":
            y += 1
        elif self.direction == "LEFT":
            x -= 1
        elif self.direction == "RIGHT":
            x += 1
        self.position = [x, y]
        self.trail.append(list(self.position)) # Add new position to trail

    def change_direction(self, new_direction):
        # Prevent reversing direction
        if new_direction == "UP" and self.direction != "DOWN":
            self.direction = "UP"
        elif new_direction == "DOWN" and self.direction != "UP":
            self.direction = "DOWN"
        elif new_direction == "LEFT" and self.direction != "RIGHT":
            self.direction = "LEFT"
        elif new_direction == "RIGHT" and self.direction != "LEFT":
            self.direction = "RIGHT"

    def handle_input(self, event):
        if event.type == pygame.KEYDOWN:
            # Example controls (adjust keys as needed)
            if self.name == "Human" or self.name == "P1": # Player 1 controls
                if event.key == pygame.K_UP:
                    self.change_direction("UP")
                elif event.key == pygame.K_DOWN:
                    self.change_direction("DOWN")
                elif event.key == pygame.K_LEFT:
                    self.change_direction("LEFT")
                elif event.key == pygame.K_RIGHT:
                    self.change_direction("RIGHT")
            elif self.name == "P2": # Player 2 controls (if applicable)
                if event.key == pygame.K_w:
                    self.change_direction("UP")
                elif event.key == pygame.K_s:
                    self.change_direction("DOWN")
                elif event.key == pygame.K_a:
                    self.change_direction("LEFT")
                elif event.key == pygame.K_d:
                    self.change_direction("RIGHT")

    def set_sprite(self, sprite):
        """Allows setting/changing the sprite after initialization."""
        self.sprite = sprite

    def get_rotated_sprite(self, cell_w, cell_h):
        """Returns the sprite rotated to match the current direction."""
        if not self.sprite:
            return None # No sprite loaded

        # Scale the sprite first
        scaled_sprite = pygame.transform.scale(self.sprite, (int(cell_w), int(cell_h)))

        # Rotate based on direction (assuming original sprite faces RIGHT)
        if self.direction == "UP":
            return pygame.transform.rotate(scaled_sprite, 90)
        elif self.direction == "DOWN":
            return pygame.transform.rotate(scaled_sprite, -90)
        elif self.direction == "LEFT":
            return pygame.transform.rotate(scaled_sprite, 180)
        else: # RIGHT
            return scaled_sprite # No rotation needed