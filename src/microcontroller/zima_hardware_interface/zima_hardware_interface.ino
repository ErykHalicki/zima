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
  Serial.println("ZIMA:INIT:HARDWARE_INTERFACE");

  // Initialize subsystems
  servoController.init();
  encoderController.init();
  dcMotorController.init();  // Added DC motor initialization

  Serial.println("Zima Hardware Interface Ready");
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
        case 8:  // DC Motor Forward
            dcMotorController.forward(static_cast<int>(value));
            break;
        case 9:  // DC Motor Reverse
            dcMotorController.reverse(static_cast<int>(value));
            break;
        case 10:  // DC Motor Stop
            dcMotorController.stop(value > 0);
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