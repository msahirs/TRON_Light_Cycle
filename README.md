# TRON Light Cycles

This project is a Python implementation of the classic TRON light cycles game, featuring both human and AI players. The game is designed to be played in a grid where players leave trails behind them, and the objective is to avoid crashing into walls or trails.

## Project Structure

```
tron-light-cycles
├── src
│   ├── main.py        # Entry point of the application
│   ├── game.py        # Game logic and state management
│   ├── ai.py          # AI player logic
│   ├── player.py      # Player representation and movement
│   └── utils.py       # Utility functions for the game
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Requirements

To run this project, you need to install the required dependencies. You can do this by running:

```
pip install -r requirements.txt
```

## Running the Game

To start the game, run the following command in your terminal:

```
python src/main.py
```

## Features

- Play against an AI or another human player.
- Dynamic game grid with collision detection.
- Simple controls for player movement.
- AI that adapts to the player's moves.

## Contributing

Feel free to fork the repository and submit pull requests for any improvements or features you would like to add!