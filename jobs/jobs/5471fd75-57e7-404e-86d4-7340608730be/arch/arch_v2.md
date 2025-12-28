SPEC_ID: a9aa53fc-070e-4453-97e3-0605614a1b5c
SPEC_HASH: 86c19f04f7d67091453114f37ce5081c2a048de86825c09a8a7d882dbf655a8b

```yaml
name: Tetris
description: A classic Tetris game implementation with falling tetrominoes, line clearing, scoring, and progressive difficulty. Players rotate and position falling blocks to complete horizontal lines.

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

actions:
  - name: StartGame
    description: Initialize a new game - clear board, reset score/level/lines, spawn first piece, set status to playing
    stateChanges:
      - board: Reset to empty 20x10 grid
      - currentPiece: Generate random tetromino at spawn position (top center)
      - nextPiece: Generate random tetromino for preview
      - score: Reset to 0
      - level: Reset to 1
      - linesCleared: Reset to 0
      - gameStatus: Set to 'playing'
      - dropInterval: Set to initial speed (1000ms)

  - name: MovePieceLeft
    description: Move current piece one cell left if not blocked
    stateChanges:
      - currentPiece: Decrease x position by 1 if valid

  - name: MovePieceRight
    description: Move current piece one cell right if not blocked
    stateChanges:
      - currentPiece: Increase x position by 1 if valid

  - name: MovePieceDown
    description: Move current piece one cell down (soft drop), awards points
    stateChanges:
      - currentPiece: Increase y position by 1 if valid
      - score: Add 1 point for soft drop

  - name: RotatePiece
    description: Rotate current piece 90 degrees clockwise with wall kick if needed
    stateChanges:
      - currentPiece: Update rotation state (0->1->2->3->0), adjust position for wall kicks

  - name: HardDrop
    description: Instantly drop piece to lowest valid position and lock
    stateChanges:
      - currentPiece: Move to lowest valid y position
      - score: Add 2 points per cell dropped
      - board: Lock piece into board
      - Triggers LockPiece action

  - name: LockPiece
    description: Lock current piece into board, check for line clears, spawn next piece
    stateChanges:
      - board: Add current piece blocks to board grid
      - currentPiece: Set to nextPiece at spawn position (top center)
      - nextPiece: Generate new random tetromino
      - Triggers CheckLines action
      - Triggers CheckGameOver action

  - name: CheckLines
    description: Check for and clear completed horizontal lines, update score using official Tetris scoring formula
    stateChanges:
      - board: Remove completed lines, shift above lines down
      - linesCleared: Add number of lines cleared
      - score: Add points using formula basePoints * (level + 1) where basePoints are 40/100/300/1200 for 1/2/3/4 lines respectively
      - level: Increase every 10 lines cleared
      - dropInterval: Decrease based on level (faster drops)

  - name: CheckGameOver
    description: Check if new piece overlaps existing blocks (game over condition)
    stateChanges:
      - gameStatus: Set to 'gameover' if spawn position blocked
      - highScore: Update if current score exceeds high score

  - name: GameTick
    description: Automatic drop triggered by game loop timer
    stateChanges:
      - currentPiece: Move down if valid, otherwise trigger LockPiece
      - lastDropTime: Update to current timestamp

  - name: PauseGame
    description: Pause the game
    stateChanges:
      - gameStatus: Set to 'paused'

  - name: ResumeGame
    description: Resume from pause
    stateChanges:
      - gameStatus: Set to 'playing'
      - lastDropTime: Reset to current time

  - name: QuitToTitle
    description: Return to title screen
    stateChanges:
      - gameStatus: Set to 'title'

  - name: SaveHighScore
    description: Persist high score to storage
    stateChanges:
      - highScore: Save to persistent storage
```