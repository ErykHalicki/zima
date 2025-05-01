#ifndef ZIMA_DCMOTOR_CONTROL_H
#define ZIMA_DCMOTOR_CONTROL_H

#include <FS_MX1508.h>
#include "config.h"

class DCMotorController {
private:
    MX1508 motorLeft;
    MX1508 motorRight;

public:
    DCMotorController() : 
        motorLeft(DCMOTOR_LEFT_PINA, DCMOTOR_LEFT_PINB),
        motorRight(DCMOTOR_RIGHT_PINA, DCMOTOR_RIGHT_PINB)
    {}

    void init() {
        // Set PWM frequency and resolution for both motors
        motorLeft.setFrequency(DCMOTOR_FREQ);
        motorLeft.setResolution(DCMOTOR_RES);
        
        motorRight.setFrequency(DCMOTOR_FREQ);
        motorRight.setResolution(DCMOTOR_RES);
    }

    void leftMotorForward(int speed) {
        speed = constrain(speed, 0, 255);
        motorLeft.motorGo(speed);
    }

    void leftMotorReverse(int speed) {
        speed = constrain(speed, 0, 255);
        motorLeft.motorGo(-speed);
    }

    void rightMotorForward(int speed) {
        speed = constrain(speed, 0, 255);
        motorRight.motorGo(speed);
    }

    void rightMotorReverse(int speed) {
        speed = constrain(speed, 0, 255);
        motorRight.motorGo(-speed);
    }

    void leftMotorStop(bool softStop = true) {
        if (softStop) {
            motorLeft.motorGo(0);
        } else {
            motorLeft.motorBrake();
        }
    }

    void rightMotorStop(bool softStop = true) {
        if (softStop) {
            motorRight.motorGo(0);
        } else {
            motorRight.motorBrake();
        }
    }
};

#endif // ZIMA_DCMOTOR_CONTROL_H
