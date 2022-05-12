#include <PinChangeInterrupt.h>

/*
1. 使用串行输入"l"校准接收机信号中位, 根据返回值对正1500, 校准采用 "-=", 即 offset_num = num - 1500
2. 后控制信号上/下限对正 1.00/-1.00
注：1. 2. 后或许更改为输入项 待定 当前采用手动输入
3. 使用串行 输入"Y" 串行返回一个 str 结构为 CH1#CH2#CH3
*/

char date;
const byte channel_pin[] = {9, 10, 11};
volatile unsigned long rising_start[] = {0, 0, 0};
volatile long channel_length[] = {0, 0, 0};
int num;
//int ledPin = 13;

void setup() {
  Serial.begin(9600);
  
  pinMode(channel_pin[0], INPUT);
  pinMode(channel_pin[1], INPUT);
  pinMode(channel_pin[2], INPUT);
//  pinMode(ledPin, OUTPUT);

  attachPinChangeInterrupt(digitalPinToPinChangeInterrupt(channel_pin[0]), onRising0, CHANGE);
  attachPinChangeInterrupt(digitalPinToPinChangeInterrupt(channel_pin[1]), onRising1, CHANGE);
  attachPinChangeInterrupt(digitalPinToPinChangeInterrupt(channel_pin[2]), onRising2, CHANGE);
}

void processPin(byte pin) {
  uint8_t trigger = getPinChangeInterruptTrigger(digitalPinToPCINT(channel_pin[pin]));

  if (trigger == RISING) {
    rising_start[pin] = micros();
  } else if (trigger == FALLING) {
    channel_length[pin] = micros() - rising_start[pin];
  }
}

void onRising0(void) {
  processPin(0);
}

void onRising1(void) {
  processPin(1);
}

void onRising2(void) {
  processPin(2);
}

// 整理通道的值并重映射
float lengthMap(int num, int offset_num = 50) {
  float returnNum;
  num -= offset_num;
  if (num <= 1000) {
    num = 1000;
  }
  else if (num >= 2000) {
    num = 2000;
  }
  else if (num <= 1560 && num >= 1440) {
//    if (num <= 1560 && num >= 1440) {
    num = 1500;
  }
  // map不能回传float 故无法映射-1~1，故映射（-100~100）/100.0
  returnNum = (map(num, 1000, 2000, -100, 100)) / 100.0;
  return returnNum;
}


void loop() {
  if (Serial.available() != 0) { 
  date= Serial.read();
 }
 if(date == 'Y' || date == 'y') {
  // lengthMap(channel_length[0], 52）   
  // 52 = 1552-1500
  // -24 = 1476-1500
  Serial.print(lengthMap(channel_length[0], -8));
  Serial.print("#");
  Serial.print(lengthMap(channel_length[1], -16));
  Serial.print("#");
  Serial.print(lengthMap(channel_length[2], -12));
  Serial.println("");
  date = 'N';
  delay(50);           
 }
 if(date == 'L' || date == 'l') {
  Serial.print(channel_length[0]);
  Serial.print("#");
  Serial.print(channel_length[1]);
  Serial.print("#");
  Serial.print(channel_length[2]);
  Serial.println("");
  date = 'N';
  delay(50);           
 }
// if (lengthMap(channel_length[2]) == 1.00){
//  digitalWrite(ledPin,HIGH);
//  }
//  else if (lengthMap(channel_length[2]) == -1.00) {
//     digitalWrite(ledPin,LOW);
//  }
//  Serial.print(lengthMap(channel_length[2]));
//  Serial.println("");
//  delay(500);
 
}
