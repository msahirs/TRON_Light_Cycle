import os, random, pickle
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential, load_model, clone_model
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Define filenames for persistent state
MODEL_WEIGHTS_FILE = "dqn_tron_weights.weights.h5" # Use weights.h5 format
REPLAY_MEMORY_FILE = "dqn_tron_memory.pkl"
EPSILON_FILE = "dqn_tron_epsilon.pkl"

class RLAgent:
    def __init__(self,
                 state_shape=(60,80,1),
                 action_size=4,
                 lr=1e-3, gamma=0.99, # Adjusted LR slightly
                 eps_start=1.0, eps_min=0.05, eps_decay=0.9998, # Slower decay
                 memory_size=10000, # Increased memory size
                 target_update_freq=20): # How often to update target network (in steps or episodes)
        self.state_shape, self.action_size = state_shape, action_size
        self.gamma = gamma
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.learning_rate = lr
        self.target_update_freq = target_update_freq
        self.train_step_counter = 0 # Counter for target network updates

        # --- Load or Initialize State ---
        self.model = self._load_or_build_model()
        self.target_model = self._build_model(self.learning_rate) # Build structure
        self.update_target_model() # Initialize target model weights

        self.memory = self._load_replay_memory(memory_size)
        self.epsilon = self._load_epsilon(eps_start)

        print(f"RLAgent Initialized. Epsilon: {self.epsilon:.4f}, Memory Size: {len(self.memory)}")

    def _load_or_build_model(self):
        """Loads model weights if they exist, otherwise builds a new model."""
        model = self._build_model(self.learning_rate) # Always build the structure
        if os.path.exists(MODEL_WEIGHTS_FILE):
            try:
                model.load_weights(MODEL_WEIGHTS_FILE)
                print(f"Loaded model weights from {MODEL_WEIGHTS_FILE}")
            except Exception as e:
                print(f"Error loading model weights: {e}. Starting with fresh weights.")
        else:
            print(f"No model weights file found at {MODEL_WEIGHTS_FILE}. Starting fresh.")
        return model

    def _load_replay_memory(self, max_size):
        """Loads replay memory from file or returns a new deque."""
        if os.path.exists(REPLAY_MEMORY_FILE):
            try:
                with open(REPLAY_MEMORY_FILE, 'rb') as f:
                    loaded_data = pickle.load(f)

                # Create a new deque with the desired maxlen, populated with loaded data
                # This handles both cases: loaded_data is already a deque or just an iterable (like a list)
                memory = deque(loaded_data, maxlen=max_size)

                print(f"Loaded replay memory ({len(memory)} items) from {REPLAY_MEMORY_FILE}")
                return memory
            except Exception as e:
                # Print a more specific error if possible
                print(f"Error loading replay memory: {e}. Starting with empty memory.")
        else:
            print(f"No replay memory file found at {REPLAY_MEMORY_FILE}. Starting fresh.")
        # Return a new, empty deque if loading failed or file didn't exist
        return deque(maxlen=max_size)

    def _load_epsilon(self, default_epsilon):
        """Loads epsilon from file or returns the default starting epsilon."""
        if os.path.exists(EPSILON_FILE):
            try:
                with open(EPSILON_FILE, 'rb') as f:
                    epsilon = pickle.load(f)
                print(f"Loaded epsilon ({epsilon:.4f}) from {EPSILON_FILE}")
                return max(self.eps_min, epsilon) # Ensure it's not below min
            except Exception as e:
                print(f"Error loading epsilon: {e}. Starting with default {default_epsilon}.")
        else:
            print(f"No epsilon file found at {EPSILON_FILE}. Starting fresh.")
        return default_epsilon

    def _build_model(self, lr):
        """Builds the Keras model structure."""
        m = Sequential([
            Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=self.state_shape),
            Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
            Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(self.action_size, activation='linear') # Q-values
        ])
        m.compile(optimizer=Adam(learning_rate=lr), loss='mse')
        print("Built new Keras model.")
        return m

    def update_target_model(self):
        """Copy weights from main model to target model."""
        print("Updating target model weights.")
        self.target_model.set_weights(self.model.get_weights())

    def get_state(self, game, player):
        """Encodes the game state from the perspective of the player."""
        # 0=empty, 1=my trail, 2=opponent trail, 3=my head, 4=opponent head
        s = np.zeros(self.state_shape, dtype=np.float32)
        my_color = player.color
        opp_colors = {p.color for p in game.players if p != player}

        for y in range(game.height):
            for x in range(game.width):
                cell_color = game.grid[y][x]
                if cell_color is not None:
                    if cell_color == my_color:
                        s[y, x, 0] = 1.0 # My trail
                    elif cell_color in opp_colors:
                        s[y, x, 0] = 2.0 # Opponent trail

        # Mark heads
        hx, hy = player.position
        if 0 <= hx < game.width and 0 <= hy < game.height:
            s[hy, hx, 0] = 3.0 # My head

        for p in game.players:
            if p is not player:
                ox, oy = p.position
                if 0 <= ox < game.width and 0 <= oy < game.height:
                    s[oy, ox, 0] = 4.0 # Opponent head
        return s

    def act(self, state, use_epsilon=True):
        """Choose action using epsilon-greedy policy."""
        if use_epsilon and random.random() < self.epsilon:
            return random.randrange(self.action_size) # Explore
        # Exploit: predict Q-values and choose the best action
        q_values = self.model.predict(state[np.newaxis, ...], verbose=0)[0]
        return int(np.argmax(q_values))

    def act_target(self, state):
        """Choose action greedily based on the target model (for the opponent)."""
        q_values = self.target_model.predict(state[np.newaxis, ...], verbose=0)[0]
        return int(np.argmax(q_values)) # Always exploit

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=128): # Increased batch size
        """Train the model using randomly sampled experiences from memory."""
        if len(self.memory) < batch_size * 2: # Wait for more samples before training
            return

        minibatch = random.sample(self.memory, batch_size)

        # Prepare batch data for faster processing
        states = np.array([transition[0] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch])

        # Predict Q-values with main model for current states
        current_q_values = self.model.predict(states, batch_size=batch_size, verbose=0)
        # Predict Q-values with BOTH models for next states
        next_q_values_main = self.model.predict(next_states, batch_size=batch_size, verbose=0) # Use main model for action selection
        next_q_values_target = self.target_model.predict(next_states, batch_size=batch_size, verbose=0) # Use target model for evaluation

        # Update Q-values based on Bellman equation
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                # DDQN modification:
                # 1. Select best action 'a' using the main model for the next state
                best_action_next = np.argmax(next_q_values_main[i])
                # 2. Evaluate that action 'a' using the target model
                target = reward + self.gamma * next_q_values_target[i][best_action_next]
                # --- Original DQN line (for comparison): ---
                # target = reward + self.gamma * np.amax(next_q_values_target[i])

            target_f = current_q_values[i] # Get the predicted Q-values for the current state
            target_f[action] = target # Update the Q-value for the action taken

        # Train the model on the updated targets
        self.model.fit(states, current_q_values, epochs=1, verbose=0, batch_size=batch_size)

        # Decay epsilon
        if self.epsilon > self.eps_min:
            self.epsilon *= self.eps_decay

        # Update target network counter and potentially update weights
        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_freq == 0:
            self.update_target_model()

    def save(self):
        """Saves the model weights, replay memory, and epsilon."""
        print("Saving RL agent state...")
        try:
            self.model.save_weights(MODEL_WEIGHTS_FILE)
            print(f"  - Saved model weights to {MODEL_WEIGHTS_FILE}")
        except Exception as e:
            print(f"  - Error saving model weights: {e}")
        try:
            with open(REPLAY_MEMORY_FILE, 'wb') as f:
                pickle.dump(self.memory, f)
            print(f"  - Saved replay memory ({len(self.memory)} items) to {REPLAY_MEMORY_FILE}")
        except Exception as e:
            print(f"  - Error saving replay memory: {e}")
        try:
            with open(EPSILON_FILE, 'wb') as f:
                pickle.dump(self.epsilon, f)
            print(f"  - Saved epsilon ({self.epsilon:.4f}) to {EPSILON_FILE}")
        except Exception as e:
            print(f"  - Error saving epsilon: {e}")