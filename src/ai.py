import random

class AI:
    def __init__(self, player, difficulty="medium"):
        self.player = player
        # Map "training" to "hard" internally for logic purposes
        self.difficulty = difficulty.lower()
        if self.difficulty == "training":
            self.effective_difficulty = "hard"
            print(f"AI {player.name} using 'training' mode (effective logic: hard)")
        else:
            self.effective_difficulty = self.difficulty
            print(f"AI {player.name} using '{self.effective_difficulty}' difficulty")

        self.directions = ["UP", "DOWN", "LEFT", "RIGHT"]
        self.opposite = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}

    def is_safe(self, x, y, game_state):
        """Checks if a grid cell (x, y) is within bounds and not occupied."""
        return 0 <= x < game_state.width and \
               0 <= y < game_state.height and \
               game_state.grid[y][x] is None

    def get_next_pos(self, x, y, direction):
        """Calculates the next position based on current position and direction."""
        if direction == "UP": y -= 1
        elif direction == "DOWN": y += 1
        elif direction == "LEFT": x -= 1
        elif direction == "RIGHT": x += 1
        return x, y

    def get_possible_moves(self, game_state):
        """
        Returns a list of valid directions the AI can move to.
        Excludes reversing direction and moves leading to immediate collision.
        """
        possible_moves = []
        current_x, current_y = self.player.position
        current_dir = self.player.direction

        for move_dir in self.directions:
            if move_dir == self.opposite[current_dir]:
                continue
            next_x, next_y = self.get_next_pos(current_x, current_y, move_dir)
            if self.is_safe(next_x, next_y, game_state):
                possible_moves.append(move_dir)

        # Check if going straight is safe, even if it wasn't a valid 'turn'
        straight_x, straight_y = self.get_next_pos(current_x, current_y, current_dir)
        if self.is_safe(straight_x, straight_y, game_state) and current_dir not in possible_moves:
             possible_moves.append(current_dir) # Add straight if safe and not already included

        return possible_moves

    def make_move(self, game_state):
        """Determines and executes the AI's next move based on effective difficulty."""
        possible_moves = self.get_possible_moves(game_state)
        current_dir = self.player.direction
        best_move = current_dir

        if not possible_moves:
            # No safe moves, pick a non-reversing direction (will likely crash)
            valid_losing_moves = [d for d in self.directions if d != self.opposite[current_dir]]
            if valid_losing_moves:
                best_move = random.choice(valid_losing_moves)
            # else: best_move remains current_dir (will crash into self/wall)

        # Use self.effective_difficulty for logic branching
        elif self.effective_difficulty == "easy":
            best_move = random.choice(possible_moves)

        elif self.effective_difficulty == "medium":
            if current_dir in possible_moves:
                best_move = current_dir
            else:
                turn_options = [m for m in possible_moves if m != current_dir]
                if turn_options:
                    best_move = random.choice(turn_options)
                elif possible_moves: # Fallback if only straight was possible but failed is_safe somehow
                     best_move = random.choice(possible_moves)

        elif self.effective_difficulty == "hard":
            best_move = self.evaluate_moves_hard(possible_moves, game_state)

        # Change direction if needed
        if best_move != self.player.direction:
            self.player.change_direction(best_move)

    def evaluate_moves_hard(self, moves, game_state):
        """Evaluates moves by checking distance to obstacles."""
        best_move = self.player.direction # Default to straight
        max_distance = -1
        current_x, current_y = self.player.position

        # Ensure current direction is considered if it's in possible moves
        current_moves = list(moves) # Create a copy to potentially add straight
        if self.player.direction not in current_moves:
             straight_x, straight_y = self.get_next_pos(current_x, current_y, self.player.direction)
             if self.is_safe(straight_x, straight_y, game_state):
                 current_moves.append(self.player.direction)

        for move in current_moves:
            dist = self.look_ahead(current_x, current_y, move, game_state)
            if dist > max_distance:
                max_distance = dist
                best_move = move
            # Add randomness for ties
            elif dist == max_distance and random.choice([True, False]):
                best_move = move

        # Fallback if all moves look equally bad (max_distance is still -1 or 0)
        if max_distance <= 0 and current_moves:
             # If straight is an option and safe, prefer it slightly
             if self.player.direction in current_moves:
                 return self.player.direction
             else:
                 return random.choice(current_moves) # Otherwise random safe move
        elif not current_moves: # No moves possible at all
             valid_losing_moves = [d for d in self.directions if d != self.opposite[self.player.direction]]
             return random.choice(valid_losing_moves) if valid_losing_moves else self.player.direction

        return best_move

    def look_ahead(self, start_x, start_y, direction, game_state, max_steps=10):
        """Looks ahead in a given direction and counts safe steps."""
        count = 0
        x, y = self.get_next_pos(start_x, start_y, direction)
        while count < max_steps and self.is_safe(x, y, game_state):
            count += 1
            x, y = self.get_next_pos(x, y, direction)
        return count