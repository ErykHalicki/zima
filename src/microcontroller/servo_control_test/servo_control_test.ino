#include <ESP32Servo.h>

// create four servo objects 
Servo servo1, servo2, servo3, servo4;

int servo1Pin = 4;
// Define throttle signal values
const int minThrottle = 500; // Minimum throttle in microseconds (1ms)
const int maxThrottle = 2500; // Maximum throttle in microseconds (2ms)

int pins[] = {4, 16, 17, 5};
float pos[] = {90,90,90,90};      // position in degrees
float goal[] = {90,90,90,90};
float speed = 0.5;
Servo motors[8];

void setup() {
  // Initialize PWM for the ESC
  for (int i = 0; i < 4; i++) {
    motors[i] = Servo();
    motors[i].setPeriodHertz(50);
    motors[i].attach(pins[i]);
  }

  Serial.begin(115200);
  while(Serial.available() > 0){
    Serial.read();
  }
  
}

int sign(int i){
  if(i<0) return -1;
  return 1;
}

const byte numChars = 64;
char receivedChars[numChars];   // an array to store the received data

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
            receivedChars[ndx] = '\0'; // terminate the string
            ndx = 0;
            newData = true;
        }
    }
}

int currMotor = 0;
void parseNewData(){
    char * strtokIndx; // this is used by strtok() as an index

    strtokIndx = strtok(receivedChars,"<");      // get the first part - the string
    currMotor=atoi(strtokIndx);

    strtokIndx = strtok(NULL, "<");
    goal[currMotor] = atof(strtokIndx);     // convert this part to a float
    newData = false;
}

void loop() {
  // Check for serial input
  recvWithEndMarker();
  if(newData)parseNewData();

  for (int i = 0; i < 4; i++) {
    motors[i].write(pos[i]);
    if(pos[i] + speed > goal[i] || pos[i] - speed < goal[i]){
      pos[i] += sign(goal[i] - pos[i]) * speed;  
    }
    else{pos[i] = goal[i];}
  }
  
  delay(10);
}