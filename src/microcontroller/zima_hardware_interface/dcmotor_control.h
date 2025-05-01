#ifndef ZIMA_DCMOTOR_CONTROL_H
#define ZIMA_DCMOTOR_CONTROL_H

#include <FS_MX1508.h>
#include "config.h"

class DCMotorController {
private:
    FS_MX1508 motorA;

public:
    DCMotorController() : 
        motorA(DCMOTOR_PINA, DCMOTOR_PINB) 
    {}

    void init() {
        // Optional: Set PWM frequency and resolution if needed
        motorA.setPWMFrequency(DCMOTOR_FREQ);
        motorA.setPWMResolution(DCMOTOR_RES);
    }

    void forward(int speed) {
        // Ensure speed is within 0-255 range
        speed = constrain(speed, 0, 255);
        motorA.drive(speed);
    }

    void reverse(int speed) {
        // Ensure speed is within 0-255 range
        speed = constrain(speed, 0, 255);
        motorA.drive(-speed);
    }

    void stop(bool softStop = true) {
        if (softStop) {
            // Gradual stop by reducing speed to 0
            motorA.drive(0);
        } else {
            // Immediate hard stop
            motorA.brake();
        }
    }
};

#endif // ZIMA_DCMOTOR_CONTROL_H