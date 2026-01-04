import pygame
import threading
import time
from typing import Optional


class PS3GamepadController:
    def __init__(self, deadzone: float = 0.15):
        self.deadzone = deadzone
        self.joystick: Optional[pygame.joystick.Joystick] = None
        self.running = False
        self.update_thread: Optional[threading.Thread] = None

        self.left_stick_x = 0.0
        self.left_stick_y = 0.0
        self.right_stick_x = 0.0
        self.right_stick_y = 0.0

        self.l1_button = 0
        self.r1_button = 0
        self.l2_trigger = 0.0
        self.r2_trigger = 0.0

        self.dpad_up = 0
        self.dpad_down = 0
        self.dpad_left = 0
        self.dpad_right = 0

        self.triangle = 0
        self.circle = 0
        self.cross = 0
        self.square = 0

        self.lock = threading.Lock()

    def start(self):
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No joystick/gamepad detected")

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

        print(f"Connected to: {self.joystick.get_name()}")
        print(f"Number of axes: {self.joystick.get_numaxes()}")
        print(f"Number of buttons: {self.joystick.get_numbuttons()}")

        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()

    def _update_loop(self):
        while self.running:
            pygame.event.pump()

            with self.lock:
                self.left_stick_x = self._apply_deadzone(self.joystick.get_axis(0))
                self.left_stick_y = self._apply_deadzone(self.joystick.get_axis(1))
                self.right_stick_x = self._apply_deadzone(self.joystick.get_axis(2))
                self.right_stick_y = self._apply_deadzone(self.joystick.get_axis(3))

                self.l2_trigger = (self.joystick.get_axis(12) + 1.0) / 2.0
                self.r2_trigger = (self.joystick.get_axis(13) + 1.0) / 2.0

                self.triangle = self.joystick.get_button(12)
                self.circle = self.joystick.get_button(13)
                self.cross = self.joystick.get_button(14)
                self.square = self.joystick.get_button(15)

                self.l1_button = self.joystick.get_button(10)
                self.r1_button = self.joystick.get_button(11)

                hat = self.joystick.get_hat(0)
                self.dpad_left = 1 if hat[0] == -1 else 0
                self.dpad_right = 1 if hat[0] == 1 else 0
                self.dpad_down = 1 if hat[1] == -1 else 0
                self.dpad_up = 1 if hat[1] == 1 else 0

            time.sleep(0.01)

    def _apply_deadzone(self, value: float) -> float:
        if abs(value) < self.deadzone:
            return 0.0
        return value

    def get_state(self) -> dict:
        with self.lock:
            return {
                'left_stick_x': self.left_stick_x,
                'left_stick_y': self.left_stick_y,
                'right_stick_x': self.right_stick_x,
                'right_stick_y': self.right_stick_y,
                'l1': self.l1_button,
                'r1': self.r1_button,
                'l2': self.l2_trigger,
                'r2': self.r2_trigger,
                'triangle': self.triangle,
                'circle': self.circle,
                'cross': self.cross,
                'square': self.square,
                'dpad_up': self.dpad_up,
                'dpad_down': self.dpad_down,
                'dpad_left': self.dpad_left,
                'dpad_right': self.dpad_right,
            }

    def stop(self):
        self.running = False
        if self.update_thread is not None:
            self.update_thread.join(timeout=1.0)
        if self.joystick is not None:
            self.joystick.quit()
        pygame.quit()
