#define ENCODER_PIN 13  // Pin for the encoder signal

// Variables for calculating speed
unsigned long lastPulseTime = 0;
unsigned long currentTime = 0;
unsigned long timeBetweenPulses = 0;
unsigned long rpm = 0;
int pulsesPerRevolution = 18;  // Adjust based on your encoder specs

// Variables for debouncing
int lastState = HIGH;
unsigned long debounceDelay = 5;  // 5ms debounce time
unsigned long lastDebounceTime = 0;

void setup() {
  Serial.begin(115200);
  pinMode(ENCODER_PIN, INPUT);
  Serial.println("Rotary Encoder Speed Test");
  Serial.println("-------------------------");
}

void loop() {
  // Read the current state of the encoder
  int reading = digitalRead(ENCODER_PIN);
  //Serial.println(reading);
  
  // If state has been stable for longer than debounce delay
  if ((millis() - lastPulseTime) > debounceDelay) {
    if (reading == HIGH && lastState == LOW) {
      // Calculate time between pulses
      currentTime = millis();
      timeBetweenPulses = currentTime - lastPulseTime;
      lastPulseTime = currentTime;
      
      // Only calculate speed if we have a reasonable time between pulses
      if (timeBetweenPulses > 0 && timeBetweenPulses < 1000) {
        // Calculate RPM: (1000 * 60) / (pulses per revolution * time between pulses in ms)
        rpm = (60000) / (pulsesPerRevolution * 2 * timeBetweenPulses);
        
        // Print the results
        Serial.print("Time between pulses (ms): ");
        Serial.print(timeBetweenPulses);
        Serial.print(" | RPM: ");
        Serial.println(rpm);
      }
      // If too much time has passed, assume the rotation has stopped
      else if (timeBetweenPulses >= 1000) {
        Serial.println("Rotation stopped or very slow");
        rpm = 0;
      }
    }
  }
  
  // Save the current state for next comparison
  lastState = reading;
  
  // Small delay to avoid excessive readings
  delay(1);
}
