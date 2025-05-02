std_msgs/Header header

# Full hardware status message that matches the format received from the ESP32
# Format: ZIMA:DATA:CLICKSLEFT=<value>,CLICKSRIGHT=<value>,SERVOS=<pos1>|<pos2>|...|<pos8>

int32 clicks_left       # Left encoder clicks
int32 clicks_right      # Right encoder clicks
float32[] servo_positions  # Array of 8 servo positions