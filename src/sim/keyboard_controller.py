from controller import Controller
from pynput import keyboard

class KeyboardController(Controller):
    def __init__(self):
        super().__init__()
        self.keys_pressed = set()
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self.listener.start()

    def _on_press(self, key):
        try:
            self.keys_pressed.add(key.char)
        except AttributeError:
            pass

    def _on_release(self, key):
        try:
            self.keys_pressed.discard(key.char)
        except AttributeError:
            pass

    def update(self, model, data):
        """Update control based on currently pressed keys."""
        self.stop()

        if 'w' in self.keys_pressed:
            self.forward()
        if 's' in self.keys_pressed:
            self.backward()
        if 'a' in self.keys_pressed:
            self.turn_left()
        if 'd' in self.keys_pressed:
            self.turn_right()

        super().update(model, data)

    def cleanup(self):
        """Stop the keyboard listener."""
        self.listener.stop()
