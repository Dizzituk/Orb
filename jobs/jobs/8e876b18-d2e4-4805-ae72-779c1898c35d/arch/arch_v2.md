SPEC_ID: c1bc6d98-8103-4757-bfa5-7506f77b52a0
SPEC_HASH: a269e9f8a0bc775d6b4bcddad5f960106e08083942e2ce57467f0eb4c1f8bc4a

# Tetris Game Architecture Document

## 1. Overview

This document describes the software architecture for a browser-based Tetris game implementation. The game follows the classic Tetris mechanics where players manipulate falling tetrominoes to complete horizontal lines on a 10x20 grid playfield.

### 1.1 Technology Stack
- **Platform:** Web Browser
- **Languages:** HTML5, CSS3, JavaScript (ES6+)
- **Rendering:** HTML5 Canvas API
- **No external dependencies required**

### 1.2 Design Goals
- Clean separation of concerns between game logic, rendering, and input handling
- Responsive and smooth gameplay at 60 FPS
- Maintainable and extensible codebase
- Classic Tetris gameplay mechanics

---

## 2. System Architecture

### 2.1 High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser Environment                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │    Input     │───▶│    Game      │───▶│   Renderer   │       │
│  │   Handler    │    │   Engine     │    │              │       │
│  └──────────────┘    └──────┬───────┘    └──────────────┘       │
│                             │                                    │
│                             ▼                                    │
│                      ┌──────────────┐                           │
│                      │    State     │                           │
│                      │   Manager    │                           │
│                      └──────────────┘                           │
│                             │                                    │
│         ┌───────────────────┼───────────────────┐               │
│         ▼                   ▼                   ▼               │
│  ┌────────────┐     ┌────────────┐      ┌────────────┐         │
│  │   Board    │     │   Piece    │      │   Score    │         │
│  │   State    │     │   State    │      │   State    │         │
│  └────────────┘     └────────────┘      └────────────┘         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Descriptions

#### 2.2.1 Game Engine
**Responsibility:** Central coordinator that manages the game loop, orchestrates component interactions, and enforces game rules.

**Key Functions:**
- Initialize and configure all subsystems
- Execute the main game loop using `requestAnimationFrame`
- Coordinate timing for piece descent based on current level
- Delegate input events to appropriate handlers
- Trigger state updates and rendering cycles

**Interface:**
```javascript
class GameEngine {
    constructor(config)
    init(): void
    start(): void
    pause(): void
    resume(): void
    restart(): void
    update(deltaTime: number): void
    gameLoop(timestamp: number): void
}
```

#### 2.2.2 State Manager
**Responsibility:** Maintains all game state and provides controlled access for reading and updating state.

**Key Functions:**
- Store and manage board grid state
- Track current and next piece
- Maintain score, level, and lines cleared
- Track game status (playing, paused, game over)
- Provide state snapshots for rendering

**Interface:**
```javascript
class StateManager {
    getBoard(): number[][]
    setBoard(board: number[][]): void
    getCurrentPiece(): Piece
    setCurrentPiece(piece: Piece): void
    getNextPiece(): Piece
    setNextPiece(piece: Piece): void
    getScore(): number
    addScore(points: number): void
    getLevel(): number
    setLevel(level: number): void
    getLines(): number
    addLines(count: number): void
    getGameStatus(): GameStatus
    setGameStatus(status: GameStatus): void
    reset(): void
}
```

#### 2.2.3 Input Handler
**Responsibility:** Captures and processes user input, translating raw keyboard events into game actions.

**Key Functions:**
- Register keyboard event listeners
- Map key codes to game actions
- Implement key repeat for held keys (left/right movement)
- Debounce rotation and hard drop inputs
- Notify Game Engine of input events

**Interface:**
```javascript
class InputHandler {
    constructor(gameEngine: GameEngine)
    init(): void
    destroy(): void
    handleKeyDown(event: KeyboardEvent): void
    handleKeyUp(event: KeyboardEvent): void
}
```

**Input Mapping:**
| Key | Action |
|-----|--------|
| Arrow Left | Move piece left |
| Arrow Right | Move piece right |
| Arrow Down | Soft drop (accelerate descent) |
| Arrow Up | Rotate piece clockwise |
| Space | Hard drop (instant placement) |
| P | Toggle pause |

#### 2.2.4 Renderer
**Responsibility:** Handles all visual output by drawing game state to HTML5 Canvas elements.

**Key Functions:**
- Render the main game board grid
- Draw locked pieces on the board
- Draw the active falling piece
- Render ghost piece (drop preview)
- Display next piece preview
- Update score/level/lines display
- Show game over and pause overlays

**Interface:**
```javascript
class Renderer {
    constructor(boardCanvas: HTMLCanvasElement, nextCanvas: HTMLCanvasElement)
    init(): void
    render(state: GameState): void
    renderBoard(board: number[][]): void
    renderPiece(piece: Piece): void
    renderGhostPiece(piece: Piece, dropY: number): void
    renderNextPiece(piece: Piece): void
    renderUI(score: number, level: number, lines: number): void
    showGameOver(finalScore: number): void
    showPause(visible: boolean): void
    clear(): void
}
```

---

## 3. Data Structures

### 3.1 Game Board
The board is represented as a 2D array of integers with dimensions 10 columns × 20 rows.

```javascript
board: number[ROWS][COLS]  // 20 rows × 10 columns
```

- Value `0`: Empty cell
- Values `1-7`: Occupied cell with piece type (determines color)

**Coordinate System:**
- Origin (0,0) is top-left
- X increases rightward (columns 0-9)
- Y increases downward (rows 0-19)

### 3.2 Tetromino Definitions
Each tetromino is defined as a 2D matrix containing the piece type number.

```javascript
const SHAPES = {
    I: [[0,0,0,0], [1,1,1,1], [0,0,0,0], [0,0,0,0]],  // 4×4 matrix
    J: [[2,0,0], [2,2,2], [0,0,0]],                    // 3×3 matrix
    L: [[0,0,3], [3,3,3], [0,0,0]],                    // 3×3 matrix
    O: [[4,4], [4,4]],                                 // 2×2 matrix
    S: [[0,5,5], [5,5,0], [0,0,0]],                   // 3×3 matrix
    T: [[0,6,0], [6,6,6], [0,0,0]],                   // 3×3 matrix
    Z: [[7,7,0], [0,7,7], [0,0,0]]                    // 3×3 matrix
};
```

### 3.3 Piece Object
```javascript
class Piece {
    type: number