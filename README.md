# zima
Autonomous robot rover, longer term project for use as a platform for testing end to end neural networks for robotic manipulation

<img width="502" alt="image" src="https://github.com/user-attachments/assets/43fa0d2b-2683-41e5-a6a8-52f394d76e2e" />

## Hardware Setup and Considerations

### ESP32 Bootup Stabilization

**Important Hardware Modification:**
- Add a 1µF capacitor between EN and GND on the ESP32
- Resolves intermittent bootup issues
- Helps stabilize the microcontroller during power-on sequence
- Prevents random boot failures or serial communication problems

#### Recommended Capacitor Specification
- Capacitance: 1 µF (microfarad)
- Voltage Rating: 6.3V or higher
- Type: Ceramic or Electrolytic

### Connection Tips
- Ensure stable power supply
- Use the recommended EN pin capacitor
- Verify all connections before powering on