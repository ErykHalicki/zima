#ifndef ZIMA_ENCODER_CONTROL_H
#define ZIMA_ENCODER_CONTROL_H

#include "config.h"

// Global variables for interrupt handling
volatile long g_clicksLeft = 0;
volatile long g_clicksRight = 0;
volatile unsigned long g_lastPulseTimeLeft = 0;
volatile unsigned long g_lastPulseTimeRight = 0;

// Interrupt handlers as plain C functions
void IRAM_ATTR handleEncoderLeft() {
    unsigned long currentTime = millis();
    if ((currentTime - g_lastPulseTimeLeft) > ENCODER_DEBOUNCE_DELAY) {
        g_clicksLeft++;
        g_lastPulseTimeLeft = currentTime;
    }
}

void IRAM_ATTR handleEncoderRight() {
    unsigned long currentTime = millis();
    if ((currentTime - g_lastPulseTimeRight) > ENCODER_DEBOUNCE_DELAY) {
        g_clicksRight++;
        g_lastPulseTimeRight = currentTime;
    }
}

class EncoderController {
public:
    void init() {
        pinMode(ENCODER_PIN_LEFT, INPUT_PULLUP);
        pinMode(ENCODER_PIN_RIGHT, INPUT_PULLUP);
        
        attachInterrupt(digitalPinToInterrupt(ENCODER_PIN_LEFT), handleEncoderLeft, FALLING);
        attachInterrupt(digitalPinToInterrupt(ENCODER_PIN_RIGHT), handleEncoderRight, FALLING);
    }

    long getLeftClicks() const {
        return g_clicksLeft;
    }

    long getRightClicks() const {
        return g_clicksRight;
    }

    void resetClicks() {
        g_clicksLeft = 0;
        g_clicksRight = 0;
    }

    // Singleton pattern
    static EncoderController& getInstance() {
        static EncoderController instance;
        return instance;
    }

private:
    // Private constructor for singleton
    EncoderController() = default;
};

#endif // ZIMA_ENCODER_CONTROL_H