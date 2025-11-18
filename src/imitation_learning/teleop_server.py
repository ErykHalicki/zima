from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from threading import Lock
import os
import logging
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'zima-teleop-secret'
socketio = SocketIO(app, cors_allowed_origins="*", logger=False, engineio_logger=False)

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

camera_frame = None
camera_lock = Lock()

control_state = {
    'forward': 0.0,
    'backward': 0.0,
    'left': 0.0,
    'right': 0.0,
    'save_episode': False,
    'discard_episode': False,
    'toggle_mode': False
}
control_lock = Lock()

def set_camera_frame(frame):
    global camera_frame
    with camera_lock:
        camera_frame = frame.copy()

def get_control_state():
    with control_lock:
        state = control_state.copy()
        control_state['save_episode'] = False
        control_state['discard_episode'] = False
        control_state['toggle_mode'] = False
        return state

def update_mode(mode):
    socketio.emit('mode_update', {'mode': mode})

def print_to_terminal(message):
    socketio.emit('terminal_output', {'message': message})

def generate_frames():
    while True:
        with camera_lock:
            if camera_frame is not None:
                ret, buffer = cv2.imencode('.jpg', camera_frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.01)

@app.route('/')
def index():
    html = '''
<!DOCTYPE html>
<html>
<head>
    <title>Zima Teleoperation</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <style>
        body {
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            background-color: #1e1e1e;
            color: #ffffff;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: #4CAF50;
        }
        .video-container {
            text-align: center;
            margin: 20px 0;
            background-color: #000;
            padding: 10px;
            border-radius: 8px;
        }
        img {
            max-width: 100%;
            height: auto;
            border: 2px solid #4CAF50;
            border-radius: 4px;
        }
        .controls {
            background-color: #2d2d2d;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }
        .control-section {
            margin: 15px 0;
        }
        .control-section h3 {
            color: #4CAF50;
            margin-bottom: 10px;
        }
        .key-info {
            display: inline-block;
            margin: 5px 10px;
            padding: 5px 10px;
            background-color: #3d3d3d;
            border-radius: 4px;
            font-family: monospace;
        }
        .key {
            background-color: #4CAF50;
            color: #000;
            padding: 3px 8px;
            border-radius: 3px;
            font-weight: bold;
        }
        .status {
            text-align: center;
            padding: 10px;
            background-color: #2d2d2d;
            border-radius: 8px;
            margin: 10px 0;
        }
        .mode {
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }
        .terminal {
            background-color: #000;
            color: #00ff00;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            height: 300px;
            overflow-y: auto;
            border: 2px solid #4CAF50;
        }
        .terminal-line {
            margin: 2px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Zima Teleoperation Interface</h1>

        <div class="status">
            <div>Mode: <span class="mode" id="mode">TRAIN</span></div>
        </div>

        <div class="video-container">
            <img src="/video_feed" alt="Camera Feed">
        </div>

        <div class="terminal" id="terminal">
            <div class="terminal-line">Terminal output will appear here...</div>
        </div>

        <div class="controls">
            <div class="control-section">
                <h3>Movement Controls</h3>
                <div class="key-info"><span class="key">W</span> Forward</div>
                <div class="key-info"><span class="key">S</span> Backward</div>
                <div class="key-info"><span class="key">A</span> Turn Left</div>
                <div class="key-info"><span class="key">D</span> Turn Right</div>
            </div>

            <div class="control-section">
                <h3>Episode Controls</h3>
                <div class="key-info"><span class="key">O</span> Save Episode</div>
                <div class="key-info"><span class="key">P</span> Discard Episode</div>
            </div>

            <div class="control-section">
                <h3>Mode Control</h3>
                <div class="key-info"><span class="key">T</span> Toggle Train/Test Mode</div>
            </div>
        </div>
    </div>

    <script>
        const socket = io();
        const keysPressed = new Set();

        socket.on('mode_update', (data) => {
            document.getElementById('mode').textContent = data.mode;
        });

        socket.on('terminal_output', (data) => {
            const terminal = document.getElementById('terminal');
            const line = document.createElement('div');
            line.className = 'terminal-line';
            line.textContent = data.message;
            terminal.appendChild(line);
            terminal.scrollTop = terminal.scrollHeight;
        });

        function sendControlState() {
            const state = {
                forward: keysPressed.has('w') ? 1.0 : 0.0,
                backward: keysPressed.has('s') ? 1.0 : 0.0,
                left: keysPressed.has('a') ? 1.0 : 0.0,
                right: keysPressed.has('d') ? 1.0 : 0.0,
                save_episode: keysPressed.has('o'),
                discard_episode: keysPressed.has('p'),
                toggle_mode: keysPressed.has('t')
            };

            fetch('/control', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(state)
            });

            if (keysPressed.has('o')) keysPressed.delete('o');
            if (keysPressed.has('p')) keysPressed.delete('p');
            if (keysPressed.has('t')) keysPressed.delete('t');
        }

        document.addEventListener('keydown', (event) => {
            const key = event.key.toLowerCase();
            if (['w', 'a', 's', 'd', 'o', 'p', 't'].includes(key)) {
                event.preventDefault();
                keysPressed.add(key);
                sendControlState();
            }
        });

        document.addEventListener('keyup', (event) => {
            const key = event.key.toLowerCase();
            if (['w', 'a', 's', 'd'].includes(key)) {
                event.preventDefault();
                keysPressed.delete(key);
                sendControlState();
            }
        });
    </script>
</body>
</html>
    '''
    return html

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control', methods=['POST'])
def control():
    global control_state
    data = request.json
    with control_lock:
        control_state['forward'] = data.get('forward', 0.0)
        control_state['backward'] = data.get('backward', 0.0)
        control_state['left'] = data.get('left', 0.0)
        control_state['right'] = data.get('right', 0.0)
        if data.get('save_episode'):
            control_state['save_episode'] = True
        if data.get('discard_episode'):
            control_state['discard_episode'] = True
        if data.get('toggle_mode'):
            control_state['toggle_mode'] = True
    return jsonify({'status': 'ok'})

def start_server():
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)

if __name__ == '__main__':
    start_server()
