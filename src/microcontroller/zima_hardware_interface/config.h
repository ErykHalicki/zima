#ifndef ZIMA_HARDWARE_CONFIG_H
#define ZIMA_HARDWARE_CONFIG_H

// Servo Motor Configuration
#define SERVO_COUNT 8
const int SERVO_PINS[] = {16, 27, 25, 14, 26, 18, 17, 19};

// Rotary Encoder Configuration
#define ENCODER_PIN_LEFT 35   // Left encoder pin
#define ENCODER_PIN_RIGHT 34  // Right encoder pin

// DC Motor Configuration
#define DCMOTOR_LEFT_PINA 22
#define DCMOTOR_LEFT_PINB 23
#define DCMOTOR_RIGHT_PINA 2
#define DCMOTOR_RIGHT_PINB 15

#define DCMOTOR_RES 8
#define DCMOTOR_FREQ 5000

// Serial Communication
#define SERIAL_BAUD 115200

// Timing and Debounce
#define ENCODER_DEBOUNCE_DELAY 5  // ms
#define MAIN_LOOP_DELAY 10        // ms

// Servo Motor Movement Speed
#define SERVO_SPEED 0.6         // Range: 0.1 - 1.0, controls servo movement speed

#endif // ZIMA_HARDWARE_CONFIG_H
