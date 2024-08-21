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

modelConfig = 'yolov3-tiny.cfg'
modelWeights = 'yolov3-tiny.weights'
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

# Function to handle all motion sensors
def handle_sensors():
    while True:
        for sensor_name, sensor in pir_sensors.items():
            if sensor.motion_detected:
                print(f"Motion Detected by {sensor_name}")
                speak(sensor_audio_messages[sensor_name])
                
                # Notify camera core to capture an image
                camera_event.set()  # Trigger camera capture
                
                # Wait for motion to stop
                sensor.wait_for_no_motion()
                print(f"Motion Stopped by {sensor_name}")

# Function to handle camera operations
def handle_camera():
    while True:
        # Wait for camera trigger from sensor core
        camera_event.wait()  # Wait until an event is set
        
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
        
        # Reset the camera event
        camera_event.clear()

# Dictionary to map sensor names to their audio messages
sensor_audio_messages = {
    "front_left": "Object detected on front left",
    "front_right": "Object detected on front right",
    "back_left": "Object detected on back left",
    "back_right": "Object detected on back right",
    "front": "Object detected on front",
    "back": "Object detected on back"
}

# Lock for camera access
camera_lock = threading.Lock()

# Event to signal camera capture
camera_event = threading.Event()

# Start sensor handling thread
sensor_thread = threading.Thread(target=handle_sensors)
sensor_thread.daemon = True
sensor_thread.start()

# Start camera handling thread
camera_thread = threading.Thread(target=handle_camera)
camera_thread.daemon = True
camera_thread.start()

# Start the ultrasonic sensor in a separate thread
ultrasonic_thread = threading.Thread(target=ultrasonic_sensor)
ultrasonic_thread.daemon = True
ultrasonic_thread.start()

# Keep the main thread running
while True:
    time.sleep(1)

    # Cleanup code if necessary (e.g., release resources)
    # Uncomment if using OpenCV's GUI
    # cv2.destroyAllWindows()
