#include <ESP32Servo.h>

// Define throttle signal values
const int minThrottle = 600; // Minimum throttle in microseconds (1ms)
const int maxThrottle = 2400; // Maximum throttle in microseconds (2ms)

int pins[] = {16, 27, 25, 14, 26, 18, 17, 19};
float pos[] = {-1,-1,-1,-1,-1,-1,-1,-1};      // position in degrees
float goal[] = {-1,-1,-1,-1,-1,-1,-1,-1};
float speed = 0.25;
Servo motors[8];

void setup() {
  // Initialize PWM for the ESC
  for (int i = 0; i < 8; i++) {
    motors[i] = Servo();
    motors[i].setPeriodHertz(50);
    //motors[i].attach(pins[i]);
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
    if(currMotor < 8){
      goal[currMotor] = atof(strtokIndx);     // convert this part to a float
      if(currMotor == 2 || currMotor == 1){ // when you set shoulder joint, it sets left and right to the same angle regardless of which one you set
        goal[2] = goal[currMotor];
        goal[1] = 180 - goal[2];
        if(goal[2] == -1 || goal[1] == -1){//if disabling, disable both
          goal[1] = -1;
          goal[2] = -1;
        }
      }

      if(currMotor == 6 || currMotor == 7){ // when you set shoulder joint, it sets left and right to the same angle regardless of which one you set
        goal[7] = goal[currMotor];
        goal[6] = 180 - goal[7];
      }

      if(currMotor == 4){
        if(goal[currMotor] != -1)
          goal[currMotor] = 180 - goal[currMotor]; //invert whatever was received at 4, this is to maintain 0 degrees as "forward"
      }
    }
    else atof(strtokIndx); //clear the command if the motor # is invalid}
    newData = false;
}

void loop() {
  // Check for serial input
  recvWithEndMarker();
  if(newData)parseNewData();
  for (int i = 0; i < 8; i++) {
    Serial.print(pos[i]);
    if(goal[i] != -1 && pos[i] == -1){
      motors[i].attach(pins[i]);
      Serial.println("attached a motor");
      pos[i] = goal[i];
    }
    if(goal[i] == -1 && pos[i] != -1){
      motors[i].detach();
      pos[i] = -1;
    }
  }
  Serial.println(" ");
  if(pos[2] != -1)
    pos[1] = 180 - pos[2]; // extra making sure the angles are correct
  for (int i = 0; i < 8; i++) {
    if(pos[i] != -1){
      motors[i].write(pos[i]);
      if(pos[i] + speed > goal[i] || pos[i] - speed < goal[i]){
        pos[i] += sign(goal[i] - pos[i]) * speed;  
      }
      else{pos[i] = goal[i];}
    }
  }
  
  delay(1);
}
