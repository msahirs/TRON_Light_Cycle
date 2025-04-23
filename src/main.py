import sys
import threading
import pygame
import pickle
import os
import numpy as np
import atexit  # For saving on exit
import random  # Make sure random is imported
import tensorflow as tf  # Import tensorflow
from game import Game
from player import Player
from ai import AI
from rl_agent import RLAgent


# --- Constants ---
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
GRID_WIDTH, GRID_HEIGHT = 70, 50
CELL_WIDTH = SCREEN_WIDTH // GRID_WIDTH
CELL_HEIGHT = SCREEN_HEIGHT // GRID_HEIGHT

# Colors
BLACK, WHITE = (0, 0, 0), (255, 255, 255)
CYAN, YELLOW, ORANGE = (0, 255, 255), (255, 255, 0), (255, 165, 0)
GRID_COLOR = (40, 40, 40)

# --- Check for GPU ---
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print(f"TensorFlow detected GPU: {gpu_devices}")
    try:
        # Prevent TensorFlow from allocating all GPU memory at once
        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Enabled memory growth for GPU.")
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}")
else:
    print("TensorFlow did not detect a GPU. Running on CPU.")

# --- Load Assets ---
try:
    cyan_sprite_img = pygame.Surface((CELL_WIDTH, CELL_HEIGHT))
    cyan_sprite_img.fill(CYAN)
    pygame.draw.rect(cyan_sprite_img, WHITE, (CELL_WIDTH * 0.7, 0, CELL_WIDTH * 0.3, CELL_HEIGHT))
    yellow_sprite_img = pygame.Surface((CELL_WIDTH, CELL_HEIGHT))
    yellow_sprite_img.fill(YELLOW)
    pygame.draw.rect(yellow_sprite_img, WHITE, (CELL_WIDTH * 0.7, 0, CELL_WIDTH * 0.3, CELL_HEIGHT))
    orange_sprite_img = pygame.Surface((CELL_WIDTH, CELL_HEIGHT))
    orange_sprite_img.fill(ORANGE)
    pygame.draw.rect(orange_sprite_img, WHITE, (CELL_WIDTH * 0.7, 0, CELL_WIDTH * 0.3, CELL_HEIGHT))
except pygame.error as e:
    print(f"Error loading sprites: {e}. Using placeholders.")
    cyan_sprite_img = pygame.Surface((10, 10))
    cyan_sprite_img.fill(CYAN)
    yellow_sprite_img = pygame.Surface((10, 10))
    yellow_sprite_img.fill(YELLOW)
    orange_sprite_img = pygame.Surface((10, 10))
    orange_sprite_img.fill(ORANGE)

cyan_sprite = cyan_sprite_img
yellow_sprite = yellow_sprite_img
orange_sprite = orange_sprite_img

# Model file
MODEL_FILE = "dqn_model.keras"

# --- Global variable to hold the agent for saving on exit ---
rl_agent_instance = None

# --- Function to save agent state on program exit ---
def cleanup_save_agent():
    global rl_agent_instance
    if rl_agent_instance:
        print("\nProgram exiting. Attempting to save RL Agent state...")
        rl_agent_instance.save()
    else:
        print("\nProgram exiting. No RL Agent instance to save.")

# Register the cleanup function to be called on exit
atexit.register(cleanup_save_agent)

def start_screen(screen):
    font_large = pygame.font.Font(None, 74)
    font_medium = pygame.font.Font(None, 50)
    mode_options = ["1 Player (vs AI)", "2 Players", "Spectate AI vs AI"]
    difficulty_options = ["Easy", "Medium", "Hard", "Training"]
    difficulties = ["easy", "medium", "hard", "training"]

    selected_mode = 0
    selected_difficulty = 1
    choosing_mode = True

    while True:
        screen.fill(BLACK)
        if choosing_mode:
            title = font_large.render("Select Mode", True, WHITE)
            screen.blit(title, title.get_rect(center=(screen.get_width() // 2, screen.get_height() // 4)))
            for i, opt in enumerate(mode_options):
                color = YELLOW if i == selected_mode else WHITE
                txt = font_medium.render(opt, True, color)
                rect = txt.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2 + i * 60))
                screen.blit(txt, rect)
        else:
            title_text = "Select AI Difficulty"
            if selected_mode == 2:
                title_text = "Select Difficulty for Both AIs"
            if difficulties[selected_difficulty] == "training":
                if selected_mode == 0:
                    title_text = "Play against Trained AI"
                elif selected_mode == 2:
                    title_text = "Run AI Training Session"

            title = font_large.render(title_text, True, WHITE)
            screen.blit(title, title.get_rect(center=(screen.get_width() // 2, screen.get_height() // 4)))
            for i, opt in enumerate(difficulty_options):
                color = YELLOW if i == selected_difficulty else WHITE
                txt = font_medium.render(opt, True, color)
                rect = txt.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2 - 30 + i * 60))
                screen.blit(txt, rect)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if choosing_mode:
                    if event.key == pygame.K_UP:
                        selected_mode = (selected_mode - 1) % len(mode_options)
                    elif event.key == pygame.K_DOWN:
                        selected_mode = (selected_mode + 1) % len(mode_options)
                    elif event.key == pygame.K_RETURN:
                        if selected_mode == 0 or selected_mode == 2:
                            choosing_mode = False
                        else:
                            return selected_mode, None
                else:
                    if event.key == pygame.K_UP:
                        selected_difficulty = (selected_difficulty - 1) % len(difficulty_options)
                    elif event.key == pygame.K_DOWN:
                        selected_difficulty = (selected_difficulty + 1) % len(difficulty_options)
                    elif event.key == pygame.K_RETURN:
                        return selected_mode, difficulties[selected_difficulty]
                    elif event.key == pygame.K_ESCAPE:
                        choosing_mode = True
        pygame.display.flip()

def init_round_for_training():
    """Initializes a game for RL self-play training (Agent vs Target)."""
    game = Game(GRID_WIDTH, GRID_HEIGHT)
    players = []
    p1_pos = [GRID_WIDTH // 4, GRID_HEIGHT // 2]
    p2_pos = [GRID_WIDTH * 3 // 4, GRID_HEIGHT // 2]

    # Player 1 is the RL Agent (current model)
    rl_player_obj = Player("RL Agent", p1_pos, "RIGHT", CYAN, cyan_sprite)
    # Player 2 is the Opponent (target model)
    opponent_player_obj = Player("RL Opponent", p2_pos, "LEFT", ORANGE, orange_sprite)

    players.extend([rl_player_obj, opponent_player_obj])

    for p in players:
        game.add_player(p)
    game.start_game()
    # Return the players specifically for easy access
    return game, rl_player_obj, opponent_player_obj

def init_round_standard(mode, difficulty):
    """Initializes a game for standard play modes."""
    game = Game(GRID_WIDTH, GRID_HEIGHT)
    players, ais = [], []
    p1_pos = [GRID_WIDTH // 4, GRID_HEIGHT // 2]
    p2_pos = [GRID_WIDTH * 3 // 4, GRID_HEIGHT // 2]

    if mode == 0:  # 1P vs AI (Rule-based or Loaded RL)
        human = Player("Human", p1_pos, "RIGHT", CYAN, cyan_sprite)
        ai_player = Player("AI", p2_pos, "LEFT", ORANGE, orange_sprite)
        players += [human, ai_player]
        # Note: RL agent logic is handled in the main loop for this mode
        if difficulty != "training":
            ais.append(AI(ai_player, difficulty=difficulty))
    elif mode == 1:  # 2P
        p1 = Player("P1", p1_pos, "RIGHT", CYAN, cyan_sprite)
        p2 = Player("P2", p2_pos, "LEFT", YELLOW, yellow_sprite)
        players += [p1, p2]
    elif mode == 2:  # Spectate AI vs AI (Rule-based only for spectating)
        a1 = Player("AI 1 (Cyan)", p1_pos, "RIGHT", CYAN, cyan_sprite)
        a2 = Player("AI 2 (Orange)", p2_pos, "LEFT", ORANGE, orange_sprite)
        players += [a1, a2]
        # Spectate mode always uses rule-based AI, difficulty selected
        ais += [AI(a1, difficulty=difficulty), AI(a2, difficulty=difficulty)]

    for p in players:
        game.add_player(p)
    game.start_game()
    return game, players, ais

def train_episode(rl: RLAgent, episode_num, screen, clock, font_small):
    """Runs one episode of RL training (Agent vs Target) and renders it."""
    game, rl_player, opponent_player = init_round_for_training()
    state_rl = rl.get_state(game, rl_player)
    state_opp = rl.get_state(game, opponent_player)

    done = False
    step = 0
    MAX_STEPS_PER_EPISODE = GRID_WIDTH * GRID_HEIGHT
    TRAINING_FPS = 500  # Control speed of visualized training

    while not done and step < MAX_STEPS_PER_EPISODE:
        step += 1

        # --- Pygame Event Handling within Training ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Saving happens via atexit
                pygame.quit()
                sys.exit()
        # ---------------------------------------------

        # 1. RL Agent chooses action
        action_rl = rl.act(state_rl, use_epsilon=True)
        dirs = ["UP", "DOWN", "LEFT", "RIGHT"]
        rl_player.change_direction(dirs[action_rl])

        # 2. Opponent Agent chooses action
        action_opp = rl.act_target(state_opp)
        opponent_player.change_direction(dirs[action_opp])

        # 3. Update game state
        game.update()

        # 4. Determine reward and next state
        next_state_rl = rl.get_state(game, rl_player)
        next_state_opp = rl.get_state(game, opponent_player)
        reward = 0
        if game.game_over:
            done = True
            # ... (Reward logic remains the same) ...
            if game.round_winner == rl_player:
                reward = 100.0
            elif game.round_winner == opponent_player:
                reward = -100.0
            else:
                reward = -50.0
        else:
            reward = -0.1

        # 5. Remember experience
        rl.remember(state_rl, action_rl, reward, next_state_rl, done)

        # Update states
        state_rl = next_state_rl
        state_opp = next_state_opp

        # 6. Perform replay
        rl.replay()

        # --- Rendering the Training Episode ---
        game.render(screen, CELL_WIDTH, CELL_HEIGHT, GRID_COLOR)
        # Display training status during the episode
        status_text = f"Training Ep: {episode_num} Step: {step} | Epsilon: {rl.epsilon:.3f}"
        status_surf = font_small.render(status_text, True, WHITE)
        screen.blit(status_surf, (10, SCREEN_HEIGHT - 30))
        pygame.display.flip()
        clock.tick(TRAINING_FPS)  # Control speed here
        # ------------------------------------

    # Print outcome at the end of the episode
    if step >= MAX_STEPS_PER_EPISODE:
        print(f"Episode {episode_num}: Reached max steps.")
    elif game.round_winner == rl_player:
        print(f"Episode {episode_num}: RL WIN! Steps: {step}")
    elif game.round_winner == opponent_player:
        print(f"Episode {episode_num}: RL LOSS! Steps: {step}")
    else:
        print(f"Episode {episode_num}: RL TIE/CRASH! Steps: {step}")

def main():
    global rl_agent_instance
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("TRON Light Cycles - RL Training")  # Update caption
    font_small = pygame.font.Font(None, 30)
    clock = pygame.time.Clock()  # Create clock here

    mode, difficulty = start_screen(screen)

    # --- Mode 1: Visual RL Self-Play Training ---
    if mode == 2 and difficulty == "training":
        print("Starting Mode: Visual RL Self-Play Training")
        # Instantiate agent (loads or starts fresh)
        rl_agent_instance = RLAgent(state_shape=(GRID_HEIGHT, GRID_WIDTH, 1))

        episode_count = 0
        # No background thread needed, run training loop directly
        while True:  # Loop indefinitely until user quits
            episode_count += 1
            # Pass screen, clock, font to train_episode for rendering
            train_episode(rl_agent_instance, episode_count, screen, clock, font_small)
            # Event handling is now inside train_episode

    # --- Mode 2: 1 Player vs Trained AI ---
    elif mode == 0 and difficulty == "training":
        print("Starting Mode: 1 Player vs Trained RL AI")
        weights_exist = os.path.exists(RLAgent.MODEL_WEIGHTS_FILE)
        rl_play_agent = RLAgent(state_shape=(GRID_HEIGHT, GRID_WIDTH, 1))
        if not weights_exist:
            print("WARNING: No trained model weights found. AI will perform randomly!")

        pygame.display.set_caption("TRON Light Cycles")

        game, players, _ = init_round_standard(mode=0, difficulty="training")
        human_player = players[0]
        rl_ai_player = players[1]
        wins = {human_player.name: 0, rl_ai_player.name: 0}
        wins_needed = 2

        match_over = False
        while not match_over:
            round_over = False
            while not round_over:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    human_player.handle_input(event)
                if not game.game_over:
                    current_rl_state = rl_play_agent.get_state(game, rl_ai_player)
                    action = rl_play_agent.act(current_rl_state, use_epsilon=False)
                    rl_ai_player.change_direction(["UP", "DOWN", "LEFT", "RIGHT"][action])
                    game.update()
                else:
                    round_over = True
                game.render(screen, CELL_WIDTH, CELL_HEIGHT, GRID_COLOR)
                win_text = f"{human_player.name}: {wins[human_player.name]}  {rl_ai_player.name}: {wins[rl_ai_player.name]}"
                text_surf = font_small.render(win_text, True, WHITE)
                screen.blit(text_surf, (10, 10))
                pygame.display.flip()
                clock.tick(15)  # Normal play speed
            if game.round_winner:
                wins[game.round_winner.name] += 1
            print(f"Round Over. Score: {wins}")
            pygame.time.wait(1500)
            if any(w >= wins_needed for w in wins.values()):
                match_over = True
            else:
                game, players, _ = init_round_standard(mode=0, difficulty="training")
                human_player = players[0]
                rl_ai_player = players[1]
        print(f"Match Over! Final Score: {wins}")
        pygame.time.wait(3000)
        pygame.quit()
        sys.exit()

    # --- Mode 3: Other Standard Modes ---
    else:
        print(f"Starting Mode: Standard Play (Mode {mode}, Difficulty {difficulty})")
        pygame.display.set_caption("TRON Light Cycles")
        game, players, ais = init_round_standard(mode=mode, difficulty=difficulty)
        wins = {}
        for p in players:
            wins[p.name] = 0
        wins_needed = 2
        match_over = False
        while not match_over:
            round_over = False
            while not round_over:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    human_players = [p for p in players if not any(ai.player == p for ai in ais)]
                    for p in human_players:
                        p.handle_input(event)
                if not game.game_over:
                    game.update()
                    for ai_agent in ais:
                        if not game.game_over:
                            ai_agent.make_move(game)
                else:
                    round_over = True
                game.render(screen, CELL_WIDTH, CELL_HEIGHT, GRID_COLOR)
                win_text = "  ".join([f"{name}: {count}" for name, count in wins.items()])
                text_surf = font_small.render(win_text, True, WHITE)
                screen.blit(text_surf, (10, 10))
                pygame.display.flip()
                clock.tick(15)  # Normal play speed
            if game.round_winner:
                wins[game.round_winner.name] += 1
            print(f"Round Over. Score: {wins}")
            pygame.time.wait(1500)
            if any(w >= wins_needed for w in wins.values()):
                match_over = True
            else:
                game, players, ais = init_round_standard(mode=mode, difficulty=difficulty)
        print(f"Match Over! Final Score: {wins}")
        pygame.time.wait(3000)
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    main()