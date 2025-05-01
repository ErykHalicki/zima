#ifndef ZIMA_HARDWARE_CONFIG_H
#define ZIMA_HARDWARE_CONFIG_H

// Servo Motor Configuration
#define SERVO_COUNT 8
const int SERVO_PINS[] = {16, 27, 25, 14, 26, 18, 17, 19};

// Rotary Encoder Configuration
#define ENCODER_PIN_LEFT 34   // Left encoder pin
#define ENCODER_PIN_RIGHT 35  // Right encoder pin

// DC Motor Configuration
#define DCMOTOR_LEFT_PINA 15
#define DCMOTOR_LEFT_PINB 2
#define DCMOTOR_RIGHT_PINA 22
#define DCMOTOR_RIGHT_PINB 23

#define DCMOTOR_RES 8
#define DCMOTOR_FREQ 5000

// Serial Communication
#define SERIAL_BAUD 115200

// Timing and Debounce
#define ENCODER_DEBOUNCE_DELAY 5  // ms
#define MAIN_LOOP_DELAY 10        // ms

#endif // ZIMA_HARDWARE_CONFIG_H
