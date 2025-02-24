#include <ESP32Servo.h>

// create four servo objects 
Servo servo1;

// Published values for SG90 servos; adjust if needed
int minUs = 500;
int maxUs = 2500;

int servo1Pin = 2;

int pos = 90;      // position in degrees

void setup() {
  Serial.begin(9600);
  servo1.setPeriodHertz(50);      // Standard 50hz servo
  servo1.attach(servo1Pin, minUs, maxUs);
}

int sign(int i){
  if(i<0) return -1;
  return 1;
}

void moveTo(int goal){
  while(pos != goal){
    servo1.write(pos);
    pos += sign(goal - pos);
    delay(10);  
  }
}

void loop() {
  if(Serial.available() > 1){
    moveTo(Serial.parseInt());
  }
  servo1.write(pos);
}
