#include <ESP32Servo.h>

// Rotary Encoder Configuration
#define ENCODER_PIN_LEFT 13   // Left encoder pin
#define ENCODER_PIN_RIGHT 14  // Right encoder pin

// Encoder Tracking Variables
unsigned long lastPulseTimeLeft = 0;
unsigned long lastPulseTimeRight = 0;
int lastStateLeft = HIGH;
int lastStateRight = HIGH;
unsigned long debounceDelay = 5;  // 5ms debounce time

// Clicks tracking
volatile long clicksLeft = 0;
volatile long clicksRight = 0;

// Servo Motor Configuration
const int minUs = 1000;
const int maxUs = 2000;
int pins[] = {16, 27, 25, 14, 26, 18, 17, 19};
float pos[] = {-1,-1,-1,-1,-1,-1,-1,-1};
float goal[] = {-1,-1,-1,-1,-1,-1,-1,-1};
float speed = 0.25;
Servo motors[8];

void IRAM_ATTR handleEncoderLeft() {
  // Interrupt handler for left encoder
  unsigned long currentTime = millis();
  if ((currentTime - lastPulseTimeLeft) > debounceDelay) {
    clicksLeft++;
    lastPulseTimeLeft = currentTime;
  }
}

void IRAM_ATTR handleEncoderRight() {
  // Interrupt handler for right encoder
  unsigned long currentTime = millis();
  if ((currentTime - lastPulseTimeRight) > debounceDelay) {
    clicksRight++;
    lastPulseTimeRight = currentTime;
  }
}

void setup() {
  // Rotary Encoder Setup
  pinMode(ENCODER_PIN_LEFT, INPUT_PULLUP);
  pinMode(ENCODER_PIN_RIGHT, INPUT_PULLUP);
  
  // Attach interrupt handlers
  attachInterrupt(digitalPinToInterrupt(ENCODER_PIN_LEFT), handleEncoderLeft, FALLING);
  attachInterrupt(digitalPinToInterrupt(ENCODER_PIN_RIGHT), handleEncoderRight, FALLING);
  
  // Servo Motor Setup
  for (int i = 0; i < 8; i++) {
    motors[i] = Servo();
    motors[i].setPeriodHertz(50);
  }

  Serial.begin(115200);
  Serial.println("ZIMA:INIT:MOTOR_ENCODER_INTEGRATED");
}

int sign(int i) {
  return (i < 0) ? -1 : 1;
}

const byte numChars = 64;
char receivedChars[numChars];
boolean newData = false;

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

int currMotor = 0;
void parseNewData() {
    char * strtokIndx;

    strtokIndx = strtok(receivedChars, "<");
    currMotor = atoi(strtokIndx);

    strtokIndx = strtok(NULL, "<");
    if (currMotor < 8) {
      goal[currMotor] = atof(strtokIndx);
      
      // Special handling for specific motor groups
      if (currMotor == 2 || currMotor == 1) {
        goal[2] = goal[currMotor];
        goal[1] = 180 - goal[2];
        if (goal[2] == -1 || goal[1] == -1) {
          goal[1] = -1;
          goal[2] = -1;
        }
      }

      if (currMotor == 6 || currMotor == 7) {
        goal[7] = goal[currMotor];
        goal[6] = 180 - goal[7];
      }

      if (currMotor == 4) {
        if (goal[currMotor] != -1)
          goal[currMotor] = 180 - goal[currMotor];
      }
    }
    newData = false;
}

void updateMotors() {
  for (int i = 0; i < 8; i++) {
    if (goal[i] != -1 && pos[i] == -1) {
      motors[i].attach(pins[i]);
      pos[i] = goal[i];
    }
    if (goal[i] == -1 && pos[i] != -1) {
      motors[i].detach();
      pos[i] = -1;
    }
  }
  
  if (pos[2] != -1)
    pos[1] = 180 - pos[2];
  
  for (int i = 0; i < 8; i++) {
    if (pos[i] != -1) {
      motors[i].write(pos[i]);
      if (pos[i] + speed > goal[i] || pos[i] - speed < goal[i]) {
        pos[i] += sign(goal[i] - pos[i]) * speed;  
      }
      else {
        pos[i] = goal[i];
      }
    }
  }
}

void reportStatus() {
  // Formatted output for ROS2 Serial Parser
  Serial.print("ZIMA:DATA:");
  
  // Encoder Data
  Serial.print("CLICKSLEFT=");
  Serial.print(clicksLeft);
  Serial.print(",CLICKSRIGHT=");
  Serial.print(clicksRight);
  Serial.print(",");
  
  // Motor Positions
  Serial.print("MOTORS=");
  for (int i = 0; i < 8; i++) {
    Serial.print(pos[i]);
    if (i < 7) Serial.print("|");
  }
  Serial.println();

  // Reset clicks after reporting (optional, depends on your use case)
  // Uncomment if you want to track incremental clicks
  // clicksLeft = 0;
  // clicksRight = 0;
}

void loop() {
  // Check for Serial Input
  recvWithEndMarker();
  if (newData) parseNewData();
  
  // Update Motor Positions
  updateMotors();
  
  // Report System Status
  reportStatus();
  
  delay(10);  // Small delay to avoid excessive processing
}