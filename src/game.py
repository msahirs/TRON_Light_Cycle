import pygame
from collections import Counter

# Helper function to darken a color
def darken_color(color, factor=0.5):
    """Returns a darker version of the input color."""
    r, g, b = color[:3] # Ignore alpha if present
    r = max(0, int(r * factor))
    g = max(0, int(g * factor))
    b = max(0, int(b * factor))
    return (r, g, b)

class Game:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.players = []
        self.grid = [[None for _ in range(width)] for _ in range(height)]
        self.game_over = False
        self.round_winner = None   # Will store the winning player for the round

    def start_game(self):
        self.game_over = False
        self.round_winner = None
        self.grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        for player in self.players:
            player.reset()
            x, y = player.position
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y][x] = player.color

    def update(self):
        if self.game_over:
            return

        new_positions = {}
        for player in self.players:
            x, y = player.position
            # simulate move
            if player.direction == "UP":
                y -= 1
            elif player.direction == "DOWN":
                y += 1
            elif player.direction == "LEFT":
                x -= 1
            elif player.direction == "RIGHT":
                x += 1
            new_positions[player] = (x, y)

        # Determine safety for each player's move
        safe_map = {}
        for player, (nx, ny) in new_positions.items():
            safe = True
            if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                safe = False
            elif self.grid[ny][nx] is not None:
                safe = False
            safe_map[player] = safe

        # For moves that are initially safe, check for head-on collisions.
        pos_counts = Counter(pos for player, pos in new_positions.items() if safe_map[player])
        for player, pos in new_positions.items():
            if safe_map[player] and pos_counts[pos] > 1:
                safe_map[player] = False

        # If not everyone is safe, round is over.
        if any(not safe for safe in safe_map.values()):
            self.game_over = True
            winners = [player for player, safe in safe_map.items() if safe]
            if len(winners) == 1:
                self.round_winner = winners[0]
            else:
                self.round_winner = None  # Tie round if none or multiple survive
            return

        # Otherwise, commit moves and leave trails.
        for player in self.players:
            old_x, old_y = player.position
            if 0 <= old_x < self.width and 0 <= old_y < self.height:
                self.grid[old_y][old_x] = player.color
            player.position = list(new_positions[player])
            
    def check_collision(self, player):
        x, y = player.position
        return x < 0 or x >= self.width or y < 0 or y >= self.height or self.grid[y][x] is not None

    def add_player(self, player):
        self.players.append(player)
        x, y = player.position
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x] = player.color

    def render(self, screen, cell_w, cell_h, grid_color):
        screen.fill((0, 0, 0))  # Black background

        # Draw dark blue grid lines
        dark_blue = (0, 0, 60)
        for x_pos in range(0, screen.get_width(), int(cell_w * 2)):
            pygame.draw.line(screen, dark_blue, (x_pos, 0), (x_pos, screen.get_height()), width=4)
        for y_pos in range(0, screen.get_height(), int(cell_h * 2)):
            pygame.draw.line(screen, dark_blue, (0, y_pos), (screen.get_width(), y_pos), width=4)

        # Draw trails with a darker (transparent-style) color
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] is not None:
                    orig = self.grid[y][x]
                    trail_color = darken_color(orig, 0.6)
                    r = pygame.Rect(x * cell_w, y * cell_h, cell_w, cell_h)
                    pygame.draw.rect(screen, trail_color, r)

        # Draw the current player positions using their sprites
        for p in self.players:
            x, y = p.position
            if 0 <= x < self.width and 0 <= y < self.height:
                rotated_sprite = p.get_rotated_sprite(cell_w, cell_h)
                if rotated_sprite:
                    blit_x = x * cell_w
                    blit_y = y * cell_h
                    rect = rotated_sprite.get_rect(topleft=(blit_x, blit_y))
                    screen.blit(rotated_sprite, rect)
                else:
                    r = pygame.Rect(x * cell_w, y * cell_h, cell_w, cell_h)
                    pygame.draw.rect(screen, p.color, r)

        if self.game_over:
            # Optional: Display Game Over message
            font = pygame.font.Font(None, 50)
            text = font.render("GAME OVER", True, (255, 0, 0))
            text_rect = text.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2))
            screen.blit(text, text_rect)

    def is_safe(self, x, y):
        """Checks if a grid cell (x, y) is within bounds and not occupied by a trail."""
        return 0 <= x < self.width and \
               0 <= y < self.height and \
               self.grid[y][x] is None

    def get_next_pos(self, x, y, direction):
        """Calculates the next position based on current position and direction."""
        # Ensure this logic matches player movement exactly
        if direction == "UP": y -= 1
        elif direction == "DOWN": y += 1
        elif direction == "LEFT": x -= 1
        elif direction == "RIGHT": x += 1
        return x, y