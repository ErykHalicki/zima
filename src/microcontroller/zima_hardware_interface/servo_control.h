#ifndef ZIMA_SERVO_CONTROL_H
#define ZIMA_SERVO_CONTROL_H

#include <ESP32Servo.h>
#include "config.h"

class ServoController {
private:
    Servo motors[SERVO_COUNT];
    float pos[SERVO_COUNT];
    float goal[SERVO_COUNT];
    float speed = SERVO_SPEED; // degrees per second
    unsigned long lastUpdateTime;

    int sign(float i) {
        return (i < 0) ? -1 : 1;
    }

public:
    ServoController() {
        for (int i = 0; i < SERVO_COUNT; i++) {
            pos[i] = -1;
            goal[i] = -1;
        }
        lastUpdateTime = micros();
    }

    void init() {
        for (int i = 0; i < SERVO_COUNT; i++) {
            motors[i] = Servo();
            motors[i].setPeriodHertz(50);
        }
    }

    void updatePositions() {
        // Calculate delta time in seconds
        unsigned long currentTime = micros();
        float deltaTime = (currentTime - lastUpdateTime) / 1000000.0; // Convert to seconds
        lastUpdateTime = currentTime;
        
        for (int i = 0; i < SERVO_COUNT; i++) {
            if (goal[i] != -1 && pos[i] == -1) {
                motors[i].attach(SERVO_PINS[i]);
                pos[i] = goal[i];
            }
            if (goal[i] == -1 && pos[i] != -1) {
                motors[i].detach();
                pos[i] = -1;
            }
        }
  
        if (pos[2] != -1)
            pos[1] = 180 - pos[2];
  
        for (int i = 0; i < SERVO_COUNT; i++) {
            if (pos[i] != -1) {
                // Calculate movement based on speed (degrees per second) and delta time
                float movement = speed * deltaTime;
                float distanceToGoal = goal[i] - pos[i];
                
                motors[i].write(pos[i]);
                if (abs(distanceToGoal) > movement) {
                    pos[i] += sign(distanceToGoal) * movement;
                }
                else {
                    pos[i] = goal[i];
                }
            }
        }
    }

    void setGoal(int motorIndex, float angle) {
        if (motorIndex < SERVO_COUNT) {
            // Special handling for specific motor groups
            if (motorIndex == 2 || motorIndex == 1) {
                goal[2] = angle;
                goal[1] = 180 - goal[2];
                if (goal[2] == -1 || goal[1] == -1) {
                    goal[1] = -1;
                    goal[2] = -1;
                }
            }
            else if (motorIndex == 6 || motorIndex == 7) {
                goal[7] = angle;
                goal[6] = 180 - goal[7];
            }
            else if (motorIndex == 3) {
                if (angle != -1)
                    goal[motorIndex] = 180 - angle; //TODO FIX BUG HERE CANNOT SEND -1 GOAL
            }
            else {
                goal[motorIndex] = angle;
            }
        }
    }

    float* getPositions() {
        return pos;
    }
};

#endif // ZIMA_SERVO_CONTROL_H
