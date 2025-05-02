#include <ESP32Servo.h>
#include "config.h"
#include "servo_control.h"
#include "encoder_control.h"
#include "dcmotor_control.h"

ServoController servoController;
EncoderController& encoderController = EncoderController::getInstance();
DCMotorController dcMotorController;

const byte numChars = 64;
char receivedChars[numChars];
boolean newData = false;


void setup() {
  Serial.begin(SERIAL_BAUD);

  // Initialize subsystems
  servoController.init();
  encoderController.init();
  dcMotorController.init();

  Serial.println("ZIMA:INIT:HARDWARE_INTERFACE");
}

void recvWithEndMarker() {
    static byte ndx = 0;
    char endMarker = '\n';
    char rc;
    
    if (Serial.available() > 0) {
        rc = Serial.read();

        if (rc != endMarker) {
            receivedChars[ndx] = rc;
            ndx++;
            if (ndx >= numChars) {
                ndx = numChars - 1;
            }
        }
        else {
            receivedChars[ndx] = '\0'; 
            ndx = 0;
            newData = true;
        }
    }
}

void parseNewData() {
    char * strtokIndx;

    strtokIndx = strtok(receivedChars, "<");
    int subsystem = atoi(strtokIndx);

    strtokIndx = strtok(NULL, "<");
    float value = atof(strtokIndx);

    switch(subsystem) {
        case 0 ... 7:  // Servo motors
            servoController.setGoal(subsystem, value);
            break;
        case 8:  // Left DC Motor Forward
            dcMotorController.leftMotorForward(static_cast<int>(value));
            break;
        case 9:  // Left DC Motor Reverse
            dcMotorController.leftMotorReverse(static_cast<int>(value));
            break;
        case 10:  // Left DC Motor Stop
            dcMotorController.leftMotorStop(value > 0);
            break;
        case 11:  // Right DC Motor Forward
            dcMotorController.rightMotorForward(static_cast<int>(value));
            break;
        case 12:  // Right DC Motor Reverse
            dcMotorController.rightMotorReverse(static_cast<int>(value));
            break;
        case 13:  // Right DC Motor Stop
            dcMotorController.rightMotorStop(value > 0);
            break;
    }
}

void reportStatus() {
    Serial.print("ZIMA:DATA:");
    
    // Encoder Data
    Serial.print("CLICKSLEFT=");
    Serial.print(encoderController.getLeftClicks());
    Serial.print(",CLICKSRIGHT=");
    Serial.print(encoderController.getRightClicks());
    Serial.print(",");
    
    // Motor Positions
    Serial.print("SERVOS=");
    float* positions = servoController.getPositions();
    for (int i = 0; i < 8; i++) {
        Serial.print(positions[i]);
        if (i < 7) Serial.print("|");
    }
    Serial.println();
}

void loop() {
    // Check for Serial Input
    recvWithEndMarker();
    if (newData) {
        parseNewData();
        newData = false;
    }
    
    // Update Motor Positions
    servoController.updatePositions();
    
    // Report System Status
    reportStatus();
    
    delay(MAIN_LOOP_DELAY);
}
