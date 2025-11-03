std_msgs/Header header

uint8 MOTOR_LEFT = 0
uint8 MOTOR_RIGHT = 1

uint8 ACTION_FORWARD = 0
uint8 ACTION_REVERSE = 1
uint8 ACTION_STOP = 2

uint8 motor_id      # Which motor (MOTOR_LEFT or MOTOR_RIGHT)
uint8 action        # Movement direction (ACTION_FORWARD, ACTION_REVERSE, ACTION_STOP)
int32 speed         # Speed value (0-255 for forward/reverse, 0 or 1 for stop)
