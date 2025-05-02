#define IRpin 13

void setup() {
  Serial.begin(9600);
  pinMode(IRpin,INPUT);
}

void  loop() {
  // put your main code here, to run repeatedly:
  int IRread = digitalRead(IRpin);
  Serial.println(IRread);
}
