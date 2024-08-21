from gpiozero import LED, MotionSensor, DistanceSensor
import subprocess
import time
import threading
import cv2
import numpy as np
from picamera2 import Picamera2

# GPIO setup for motion sensors and LED
green_led = LED(17)
pir_sensors = {
    "front_left": MotionSensor(4),
    "front_right": MotionSensor(5),
    "back_left": MotionSensor(6),
    "back_right": MotionSensor(13),
    "front": MotionSensor(19),
    "back": MotionSensor(27)
}
ultrasonic = DistanceSensor(echo=24, trigger=23)

# Picamera2 setup
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration())
picam2.start()

# YOLO model setup
whT = 320
confThreshold = 0.5
nmsThreshold = 0.3
classesfile = 'coco.names'
classNames = []

with open(classesfile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

modelConfig = 'yolov3.cfg'
modelWeights = 'yolov3.weights'
net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Function to use espeak to speak the provided text
def speak(text):
    subprocess.run(['espeak', text])

# Function to find objects in the camera frame
def findObject(outputs, im):
    hT, wT, cT = im.shape
    bbox = []
    classIds = []
    confs = []
    identified_objects = []
    
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
    
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
   
    for i in indices:
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        object_name = classNames[classIds[i]]
        identified_objects.append(object_name)
        
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(im, f'{object_name.upper()} {int(confs[i] * 100)}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    return identified_objects

# Function to handle ultrasonic sensor output every 10 seconds
def ultrasonic_sensor():
    while True:
        distance = ultrasonic.distance * 100
        # Uncomment to use voice feedback for distance
        speak(f"Object detected at a distance of {distance:.2f} centimeters")
        time.sleep(10)

# Start the ultrasonic sensor in a separate thread
ultrasonic_thread = threading.Thread(target=ultrasonic_sensor)
ultrasonic_thread.daemon = True
ultrasonic_thread.start()

# Dictionary to map sensor names to their audio messages
sensor_audio_messages = {
    "front_left": "Object detected on front left",
    "front_right": "Object detected on front right",
    "back_left": "Object detected on back left",
    "back_right": "Object detected on back right",
    "front": "Object detected on front",
    "back":"Object detected on back"
}

# Locks for thread-safe operations
led_lock = threading.Lock()
camera_lock = threading.Lock()

def handle_sensor(sensor_name, sensor):
    while True:
        sensor.wait_for_motion()
        print(f"Motion Detected by {sensor_name}")
        speak(sensor_audio_messages[sensor_name])
        
        # Ensure camera is accessed in a thread-safe manner
        with camera_lock:
            # Capture image from camera
            im = picam2.capture_array()
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            
            # YOLO object detection
            blob = cv2.dnn.blobFromImage(im, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
            net.setInput(blob)
            layernames = net.getLayerNames()
            outputNames = [layernames[i - 1] for i in net.getUnconnectedOutLayers()]
            
            outputs = net.forward(outputNames)
            identified_objects = findObject(outputs, im)
            
            # If objects are identified, speak their names
            if identified_objects:
                for obj in identified_objects:
                    text = f'{obj} detected'
                    speak(text)
                    time.sleep(0.5)  # Slight delay between object detections
            
            # Display the image with detected objects
            cv2.imshow('Image', im)
            cv2.waitKey(1)
        
        # Turn on LED
        with led_lock:
            green_led.on()
        
        # Wait for no motion and turn off LED
        sensor.wait_for_no_motion()
        with led_lock:
            green_led.off()
        print(f"Motion Stopped by {sensor_name}")

# Start a thread for each motion sensor
sensor_threads = []
for sensor_name, sensor in pir_sensors.items():
    thread = threading.Thread(target=handle_sensor, args=(sensor_name, sensor))
    thread.daemon = True
    sensor_threads.append(thread)
    thread.start()

# Keep the main thread running
while True:
    time.sleep(1)

    # Cleanup code if necessary (e.g., release resources)
    # Uncomment if using OpenCV's GUI
    # cv2.destroyAllWindows()
