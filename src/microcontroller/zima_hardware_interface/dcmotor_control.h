#ifndef ZIMA_DCMOTOR_CONTROL_H
#define ZIMA_DCMOTOR_CONTROL_H

#include <ESP32Servo.h>
#include "config.h"

class DCMotorController {
private:
    ESP32PWM pwmLeft1;
    ESP32PWM pwmLeft2;
    ESP32PWM pwmRight1;
    ESP32PWM pwmRight2;
    
    int leftPin1;
    int leftPin2;
    int rightPin1;
    int rightPin2;
    
    int maxPwmValue;

public:
    DCMotorController() : 
        leftPin1(DCMOTOR_LEFT_PINA),
        leftPin2(DCMOTOR_LEFT_PINB),
        rightPin1(DCMOTOR_RIGHT_PINA),
        rightPin2(DCMOTOR_RIGHT_PINB)
    {
        // Calculate max PWM value based on resolution
        maxPwmValue = (1 << DCMOTOR_RES) - 1; // 2^DCMOTOR_RES - 1
    }

    void init() {
        // Set up PWM frequency and resolution for all motor pins
        pwmLeft1.attachPin(leftPin1, DCMOTOR_FREQ, DCMOTOR_RES);
        pwmLeft2.attachPin(leftPin2, DCMOTOR_FREQ, DCMOTOR_RES);
        pwmRight1.attachPin(rightPin1, DCMOTOR_FREQ, DCMOTOR_RES);
        pwmRight2.attachPin(rightPin2, DCMOTOR_FREQ, DCMOTOR_RES);
        
        // Initialize all PWM to 0
        pwmLeft1.write(0);
        pwmLeft2.write(0);
        pwmRight1.write(0);
        pwmRight2.write(0);
    }

    void leftMotorForward(int speed) {
        speed = constrain(speed, 0, 255);
        // Map 0-255 range to max PWM value
        int pwmValue = map(speed, 0, 255, 0, maxPwmValue);
        pwmLeft1.write(pwmValue);
        pwmLeft2.write(0);
    }

    void leftMotorReverse(int speed) {
        speed = constrain(speed, 0, 255);
        // Map 0-255 range to max PWM value
        int pwmValue = map(speed, 0, 255, 0, maxPwmValue);
        pwmLeft1.write(0);
        pwmLeft2.write(pwmValue);
    }

    void rightMotorForward(int speed) {
        speed = constrain(speed, 0, 255);
        // Map 0-255 range to max PWM value
        int pwmValue = map(speed, 0, 255, 0, maxPwmValue);
        pwmRight1.write(pwmValue);
        pwmRight2.write(0);
    }

    void rightMotorReverse(int speed) {
        speed = constrain(speed, 0, 255);
        // Map 0-255 range to max PWM value
        int pwmValue = map(speed, 0, 255, 0, maxPwmValue);
        pwmRight1.write(0);
        pwmRight2.write(pwmValue);
    }

    void leftMotorStop(bool softStop = true) {
        if (softStop) {
            // Soft stop - just set PWM to 0
            pwmLeft1.write(0);
            pwmLeft2.write(0);
        } else {
            // Hard brake - set both channels to max
            pwmLeft1.write(maxPwmValue);
            pwmLeft2.write(maxPwmValue);
        }
    }

    void rightMotorStop(bool softStop = true) {
        if (softStop) {
            // Soft stop - just set PWM to 0
            pwmRight1.write(0);
            pwmRight2.write(0);
        } else {
            // Hard brake - set both channels to max
            pwmRight1.write(maxPwmValue);
            pwmRight2.write(maxPwmValue);
        }
    }
};

#endif // ZIMA_DCMOTOR_CONTROL_H
