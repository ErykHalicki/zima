std_msgs/Header header

# Command format: subsystem<value where:
# Subsystem: 
#   0-7: Servo motors (value: position in degrees)
#   8: Left DC Motor Forward (value: speed 0-255)
#   9: Left DC Motor Reverse (value: speed 0-255)
#   10: Left DC Motor Stop (value: 0 or 1)
#   11: Right DC Motor Forward (value: speed 0-255)
#   12: Right DC Motor Reverse (value: speed 0-255)
#   13: Right DC Motor Stop (value: 0 or 1)

uint8 subsystem     # The subsystem ID (0-13)
float32 value       # The command value