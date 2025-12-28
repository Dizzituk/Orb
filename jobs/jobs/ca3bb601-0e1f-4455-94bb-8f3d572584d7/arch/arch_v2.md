SPEC_ID: 15ef6ca5-05a0-4d47-b7d4-8bcf345ea32f
SPEC_HASH: 239d6e7572ad1c884b8771d3c9059be7368244297415b68a19b8811729f13c23

# Tetris Game Implementation

I'll create a complete Tetris game implementation in TypeScript with HTML5 Canvas rendering.

## Architecture Overview

The game will follow a clean separation of concerns:
- **Game Logic**: Core Tetris mechanics (pieces, rotation, collision, line clearing)
- **Rendering**: Canvas-based display with dedicated Renderer class
- **Input Handling**: Keyboard event system with action mapping
- **Game Loop**: RequestAnimationFrame-based timing with delta time tracking

---

```typescript
// src/types.ts
export type TetrominoType = 'I' | 'O' | 'T' | 'S' | 'Z' | 'J' | 'L';

export type GameAction = 
  | 'MOVE_LEFT'
  | 'MOVE_RIGHT'
  | 'MOVE_DOWN'
  | 'ROTATE_CW'
  | 'ROTATE_CCW'
  | 'HARD_DROP'
  | 'PAUSE'
  | 'RESTART';

export interface Position {
  x: number;
  y: number;
}

export interface Tetromino {
  type: TetrominoType;
  shape: number[][];
  position: Position;
  color: string;
}

export interface GameState {
  board: (string | null)[][];
  currentPiece: Tetromino | null;
  nextPiece: Tetromino | null;
  score: number;
  level: number;
  linesCleared: number;
  isGameOver: boolean;
  isPaused: boolean;
}

export interface GameConfig {
  boardWidth: number;
  boardHeight: number;
  cellSize: number;
  initialDropInterval: number;
}
```

---

```typescript
// src/constants.ts
import { GameConfig, TetrominoType, GameAction } from './types';

export const DEFAULT_CONFIG: GameConfig = {
  boardWidth: 10,
  boardHeight: 20,
  cellSize: 30,
  initialDropInterval: 1000,
};

export const TETROMINO_SHAPES: Record<TetrominoType, number[][]> = {
  I: [
    [0, 0, 0, 0],
    [1, 1, 1, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
  ],
  O: [
    [1, 1],
    [1, 1],
  ],
  T: [
    [0, 1, 0],
    [1, 1, 1],
    [0, 0, 0],
  ],
  S: [
    [0, 1, 1],
    [1, 1, 0],
    [0, 0, 0],
  ],
  Z: [
    [1, 1, 0],
    [0, 1, 1],
    [0, 0, 0],
  ],
  J: [
    [1, 0, 0],
    [1, 1, 1],
    [0, 0, 0],
  ],
  L: [
    [0, 0, 1],
    [1, 1, 1],
    [0, 0, 0],
  ],
};

export const TETROMINO_COLORS: Record<TetrominoType, string> = {
  I: '#00f0f0',
  O: '#f0f000',
  T: '#a000f0',
  S: '#00f000',
  Z: '#f00000',
  J: '#0000f0',
  L: '#f0a000',
};

export const POINTS_PER_LINE: Record<number, number> = {
  1: 100,
  2: 300,
  3: 500,
  4: 800,
};

export const LINES_PER_LEVEL = 10;

export const KEY_BINDINGS: Record<string, GameAction> = {
  'ArrowLeft': 'MOVE_LEFT',
  'ArrowRight': 'MOVE_RIGHT',
  'ArrowDown': 'MOVE_DOWN',
  'ArrowUp': 'ROTATE_CW',
  'KeyZ': 'ROTATE_CCW',
  'KeyX': 'ROTATE_CW',
  'Space': 'HARD_DROP',
  'KeyP': 'PAUSE',
  'Escape': 'PAUSE',
  'KeyR': 'RESTART',
};
```

---

```typescript
// src/tetromino.ts
import { Tetromino, TetrominoType, Position } from './types';
import { TETROMINO_SHAPES, TETROMINO_COLORS } from './constants';

const TETROMINO_TYPES: TetrominoType[] = ['I', 'O', 'T', 'S', 'Z', 'J', 'L'];

export function createTetromino(type: TetrominoType, position: Position): Tetromino {
  return {
    type,
    shape: TETROMINO_SHAPES[type].map(row => [...row]),
    position: { ...position },
    color: TETROMINO_COLORS[type],
  };
}

export function getRandomTetrominoType(): TetrominoType {
  const index = Math.floor(Math.random() * TETROMINO_TYPES.length);
  return TETROMINO_TYPES[index];
}

export function rotateTetromino(tetromino: Tetromino, clockwise: boolean = true): Tetromino {
  const shape = tetromino.shape;
  const size = shape.length;
  const rotated: number[][] = Array.from({ length: size }, () => Array(size).fill(0));

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      if (clockwise) {
        rotated[x][size - 1 - y] = shape[y][x];
      } else {
        rotated[size - 1 - x][y] = shape[y][x];
      }
    }
  }

  return {
    ...tetromino,
    shape: rotated,
  };
}

export function getTetrominoCells(tetromino: Tetromino): Position[] {
  const cells: Position[] = [];
  const { shape, position } = tetromino;

  for (let y = 0; y < shape.length; y++) {
    for (let x = 0; x < shape[y].length; x++) {
      if (shape[y][x]) {
        cells.push({
          x: position.x + x,
          y: position.y + y,
        });
      }
    }
  }

  return cells;
}
```

---

```typescript
// src/board.ts
import { GameConfig, Tetromino, Position } from './types';
import { getTetrominoCells } from './tetromino';

export function createBoard(config: GameConfig): (string | null)[][] {
  return Array.from({ length: config.boardHeight }, () =>
    Array(config.boardWidth).fill(null)
  );
}

export function isValidPosition(
  board: (string | null)[][],
  tetromino: Tetromino,
  config: GameConfig
): boolean {
  const cells = getTetrominoCells(tetromino);

  for (const cell of cells) {
    if (cell.x < 0 || cell.x >= config.boardWidth) {
      return false;
    }
    if (cell.y >= config.boardHeight) {
      return false;
    }
    if (cell.y >= 0 && board[cell.y][cell.x] !== null) {
      return false;
    }
  }