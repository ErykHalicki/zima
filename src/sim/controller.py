import numpy as np

class Controller:
    def __init__(self):
        self.left_speed = 0.0
        self.right_speed = 0.0
        self.max_speed = 1.0

    def update(self, model, data):
        """Update the robot control based on current state."""
        max_val = max(abs(self.left_speed), abs(self.right_speed))
        if max_val > self.max_speed:
            self.left_speed = self.left_speed / max_val * self.max_speed
            self.right_speed = self.right_speed / max_val * self.max_speed

        if model.nu >= 2:
            data.actuator('motor_l_wheel').ctrl = self.left_speed
            data.actuator('motor_r_wheel').ctrl = self.right_speed

    def forward(self):
        self.left_speed += self.max_speed
        self.right_speed += self.max_speed

    def backward(self):
        self.left_speed += -self.max_speed
        self.right_speed += -self.max_speed

    def turn_left(self):
        self.left_speed += -self.max_speed * 0.5
        self.right_speed += self.max_speed * 0.5

    def turn_right(self):
        self.left_speed += self.max_speed * 0.5
        self.right_speed += -self.max_speed * 0.5

    def stop(self):
        self.left_speed = 0.0
        self.right_speed = 0.0
