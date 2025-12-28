SPEC_ID: a9aa53fc-070e-4453-97e3-0605614a1b5c
SPEC_HASH: 86c19f04f7d67091453114f37ce5081c2a048de86825c09a8a7d882dbf655a8b

```yaml
name: Tetris
description: A classic Tetris game implementation with falling tetrominoes, line clearing, scoring, and progressive difficulty. Players rotate and position falling blocks to complete horizontal lines. Implements the Super Rotation System (SRS) and follows the 2009 Tetris Guideline scoring system.

screens:
  - name: TitleScreen
    description: Main menu screen with game title and start option
    components:
      - name: GameTitle
        type: text
        description: Large "TETRIS" title text centered at top
      - name: StartButton
        type: button
        description: Button to start a new game
      - name: HighScoreDisplay
        type: text
        description: Shows the current high score from storage

  - name: GameScreen
    description: Main gameplay screen with the tetris board and game info
    components:
      - name: GameBoard
        type: container
        description: 10x20 grid where tetrominoes fall and stack
      - name: CurrentPiece
        type: container
        description: The actively falling tetromino that player controls
      - name: NextPiecePreview
        type: container
        description: Shows the next tetromino that will appear
      - name: ScoreDisplay
        type: text
        description: Current score value
      - name: LevelDisplay
        type: text
        description: Current level number
      - name: LinesDisplay
        type: text
        description: Total lines cleared count
      - name: PauseButton
        type: button
        description: Button to pause the game

  - name: PauseScreen
    description: Overlay shown when game is paused
    components:
      - name: PausedText
        type: text
        description: "PAUSED" text indicator
      - name: ResumeButton
        type: button
        description: Button to resume gameplay
      - name: QuitButton
        type: button
        description: Button to quit to title screen

  - name: GameOverScreen
    description: Screen shown when game ends
    components:
      - name: GameOverText
        type: text
        description: "GAME OVER" message
      - name: FinalScoreDisplay
        type: text
        description: Shows the final score achieved
      - name: NewHighScoreText
        type: text
        description: Displayed if player achieved a new high score
      - name: PlayAgainButton
        type: button
        description: Button to start a new game
      - name: MainMenuButton
        type: button
        description: Button to return to title screen

state:
  - name: board
    type: array
    description: 2D array (20 rows x 10 cols) representing locked blocks on the board. Each cell contains null or a color string.
  - name: currentPiece
    type: object
    description: "Current falling tetromino with properties: type (I/O/T/S/Z/J/L), rotation (0-3), x position, y position, color"
  - name: nextPiece
    type: object
    description: The next tetromino that will spawn after current piece locks
  - name: score
    type: number
    description: Current game score
  - name: level
    type: number
    description: Current difficulty level (starts at 1)
  - name: linesCleared
    type: number
    description: Total number of lines cleared
  - name: gameStatus
    type: string
    description: "Current game state: 'title', 'playing', 'paused', 'gameover'"
  - name: highScore
    type: number
    description: Highest score achieved, persisted to storage
  - name: dropInterval
    type: number
    description: Milliseconds between automatic piece drops (decreases with level)
  - name: lastDropTime
    type: number
    description: Timestamp of last automatic drop for game loop timing
  - name: lastClearWasBackToBack
    type: boolean
    description: Tracks if the previous line clear was a difficult clear (Tetris or T-Spin) for back-to-back bonus calculation

rotationSystem:
  name: Super Rotation System (SRS)
  description: |
    The Super Rotation System defines piece orientations and wall kick behavior per the 2009 Tetris Guideline.
    Each piece has 4 rotation states (0, 1, 2, 3) representing 0°, 90° CW, 180°, and 270° CW.
    When a rotation would cause collision, wall kick offsets are tested in order until one succeeds or all fail.

  pieceDefinitions:
    description: |
      Each piece is defined on a grid with coordinates relative to the piece's center of rotation.
      Coordinates are (x, y) where positive x is right and positive y is down.
    
    I:
      color: cyan
      states:
        0: [[0,1], [1,1], [2,1], [3,1]]
        1: [[2,0], [2,1], [2,2], [2,3]]
        2: [[0,2], [1,2], [2,2], [3,2]]
        3: [[1,0], [1,1], [1,2], [1,3]]
    
    O:
      color: yellow
      states:
        0: [[1,0], [2,0], [1,1], [2,1]]
        1: [[1,0], [2,0], [1,1], [2,1]]
        2: [[1,0], [2,0], [1,1], [2,1]]
        3: [[1,0], [2,0], [1,1], [2,1]]
    
    T:
      color: purple
      states:
        0: [[1,0], [0,1], [1,1], [2,1]]
        1: [[1,0], [1,1], [2,1], [1,2]]
        2: [[0,1], [1,1], [2,1], [1,2]]
        3: [[1,0], [0,1], [1,1], [1,2]]
    
    S:
      color: green
      states:
        0: [[1,0], [2,0], [0,1], [1,1]]
        1: [[1,0], [1,1], [2,1], [2,2]]
        2: [[1,1], [2,1], [0,2], [1,2]]
        3: [[0,0], [0,1], [1,1], [1,2]]
    
    Z:
      color: red
      states:
        0: [[0,0], [1,0], [1,1], [2,1]]
        1: [[2,0], [1,1], [2,1], [1,2]]
        2: [[0,1], [1,1], [1,2], [2,2]]
        3: [[1,0], [0,1], [1,1], [0,2]]
    
    J:
      color: blue
      states:
        0: [[0,0], [0,1], [1,1], [2,1]]
        1: [[1,0], [2,0], [1,1], [1,2]]
        2: [[0,1], [1,1], [2,1], [2,2]]
        3: [[1,0], [1,1], [0,2], [1,2]]
    
    L:
      color: orange
      states:
        0: [[2,0], [0,1], [1,1], [2,1]]
        1: [[1,0], [1,1], [1,2], [2,2]]
        2: [[0,1], [1,1], [2,1], [0,2]]
        3: [[0,0], [1,0], [1,1], [1,2]]

  wallKickData:
    description: |
      Wall kick offsets are tested in order when rotation causes collision.
      Format: [dx, dy] where positive dx is right,