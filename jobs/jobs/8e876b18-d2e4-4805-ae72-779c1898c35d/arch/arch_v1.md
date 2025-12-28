SPEC_ID: c1bc6d98-8103-4757-bfa5-7506f77b52a0
SPEC_HASH: a269e9f8a0bc775d6b4bcddad5f960106e08083942e2ce57467f0eb4c1f8bc4a

# Tetris Game Implementation

I'll create a complete Tetris game in HTML/CSS/JavaScript.

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tetris</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        .game-container {
            display: flex;
            gap: 20px;
            padding: 20px;
        }

        .main-board {
            background: #0a0a0a;
            border: 4px solid #444;
            border-radius: 8px;
            box-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
        }

        .side-panel {
            display: flex;
            flex-direction: column;
            gap: 20px;
            color: #fff;
        }

        .info-box {
            background: rgba(0, 0, 0, 0.7);
            border: 2px solid #444;
            border-radius: 8px;
            padding: 15px;
            min-width: 150px;
        }

        .info-box h3 {
            color: #0ff;
            margin-bottom: 10px;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 2px;
        }

        .info-box .value {
            font-size: 24px;
            font-weight: bold;
            color: #fff;
        }

        .next-piece-canvas {
            background: #0a0a0a;
            border-radius: 4px;
        }

        .controls {
            font-size: 12px;
            line-height: 1.8;
            color: #888;
        }

        .controls span {
            color: #0ff;
        }

        .game-over-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            display: none;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            z-index: 100;
        }

        .game-over-overlay.show {
            display: flex;
        }

        .game-over-overlay h1 {
            color: #ff0040;
            font-size: 48px;
            margin-bottom: 20px;
            text-shadow: 0 0 20px #ff0040;
        }

        .game-over-overlay p {
            color: #fff;
            font-size: 24px;
            margin-bottom: 30px;
        }

        .game-over-overlay button {
            background: #0ff;
            color: #000;
            border: none;
            padding: 15px 40px;
            font-size: 18px;
            font-weight: bold;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.3s;
        }

        .game-over-overlay button:hover {
            background: #fff;
            transform: scale(1.05);
        }

        .pause-indicator {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #0ff;
            font-size: 48px;
            font-weight: bold;
            text-shadow: 0 0 20px #0ff;
            display: none;
            z-index: 50;
        }

        .pause-indicator.show {
            display: block;
        }
    </style>
</head>
<body>
    <div class="game-container">
        <canvas id="board" class="main-board"></canvas>
        <div class="side-panel">
            <div class="info-box">
                <h3>Score</h3>
                <div class="value" id="score">0</div>
            </div>
            <div class="info-box">
                <h3>Level</h3>
                <div class="value" id="level">1</div>
            </div>
            <div class="info-box">
                <h3>Lines</h3>
                <div class="value" id="lines">0</div>
            </div>
            <div class="info-box">
                <h3>Next</h3>
                <canvas id="next" class="next-piece-canvas"></canvas>
            </div>
            <div class="info-box controls">
                <h3>Controls</h3>
                <span>←→</span> Move<br>
                <span>↑</span> Rotate<br>
                <span>↓</span> Soft Drop<br>
                <span>Space</span> Hard Drop<br>
                <span>P</span> Pause
            </div>
        </div>
    </div>

    <div class="game-over-overlay" id="gameOver">
        <h1>GAME OVER</h1>
        <p>Final Score: <span id="finalScore">0</span></p>
        <button onclick="game.restart()">Play Again</button>
    </div>

    <div class="pause-indicator" id="pauseIndicator">PAUSED</div>

    <script>
        const COLS = 10;
        const ROWS = 20;
        const BLOCK_SIZE = 30;
        const COLORS = [
            null,
            '#00f0f0', // I - Cyan
            '#0000f0', // J - Blue
            '#f0a000', // L - Orange
            '#f0f000', // O - Yellow
            '#00f000', // S - Green
            '#a000f0', // T - Purple
            '#f00000'  // Z - Red
        ];

        const SHAPES = [
            null,
            [[0,0,0,0], [1,1,1,1], [0,0,0,0], [0,0,0,0]], // I
            [[2,0,0], [2,2,2], [0,0,0]],                   // J
            [[0,0,3], [3,3,3], [0,0,0]],                   // L
            [[4,4], [4,4]],                                 // O
            [[0,5,5], [5,5,0], [0,0,0]],                   // S
            [[0,6,0], [6,6,6], [0,0,0]],                   // T
            [[7,7,0], [0,7,7], [0,0,0]]                    // Z
        ];

        class Piece {
            constructor(type) {
                this.type = type;
                this.shape = SHAPES[type].map(row => [...row]);
                this.color = COLORS[type];
                this.x = Math.floor(COLS / 2) - Math.floor(this.shape[0].length / 2);
                this.y = 0;
            }

            rotate() {
                const rotated = [];
                const rows = this.shape.length;
                const cols = this.shape[