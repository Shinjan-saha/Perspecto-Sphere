from gpiozero import LED, MotionSensor, DistanceSensor
import subprocess
import time

green_led = LED(17)
pir = MotionSensor(4)

ultrasonic = DistanceSensor(echo=24, trigger=23)


green_led.off()

def speak(text):
    """Function to use espeak to speak the provided text."""
    subprocess.run(['espeak', text])

while True:
   
    pir.wait_for_motion()
    print("Motion Detected")
    
   
    distance = ultrasonic.distance * 100  
    print(f"Distance from object: {distance:.2f} cm")
    
  
    if distance < 200: 
        green_led.on()
        speak(f"Object detected at a distance of {distance:.2f} centimeters")
        
        
        pir.wait_for_no_motion()
        green_led.off()
        print("Motion Stopped")
    
  
    time.sleep(1)
