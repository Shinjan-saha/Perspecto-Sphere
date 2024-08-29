import time
import threading
import cv2
import numpy as np
from gpiozero import LED, MotionSensor, DistanceSensor, InputDevice
import subprocess
from picamera2 import Picamera2

# GPIO setup for motion sensors and LED
green_led = LED(17)
control_pin = InputDevice(16)  # Changed to InputDevice for control pin

pir_sensors = {
    "front_left": MotionSensor(4),
    "front_right": MotionSensor(5),
    "back_left": MotionSensor(6),
    "back_right": MotionSensor(13),
    "front": MotionSensor(19),
    "back": MotionSensor(26)
}
ultrasonic = DistanceSensor(echo=24, trigger=23)

# Picamera2 setup
picam2 = Picamera2()
main = {'size': (1024, 768)}
raw = {'size': (1024, 768)}
controls = {'FrameRate': 60, 'NoiseReductionMode': 3}
config = picam2.create_video_configuration(main, raw=raw, controls=controls)
picam2.configure(config)
picam2.start()

# YOLO model setup
whT = 320
confThreshold = 0.5
nmsThreshold = 0.3
classesfile = 'coco.names'
classNames = []

with open(classesfile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

modelConfig = 'yolov7-tiny.cfg'
modelWeights = 'yolov7-tiny.weights'
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
        
        # Draw rectangle and label on detected objects
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(im, f'{object_name.upper()} {int(confs[i] * 100)}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        time.sleep(0.5)
    
    return identified_objects

# Function to handle ultrasonic sensor output every 10 seconds
def ultrasonic_sensor():
    while True:
        distance = ultrasonic.distance * 100  
        #speak(f"Object detected at a distance of {distance:.2f} centimeters")
        time.sleep(10)

# Start the ultrasonic sensor in a separate thread
ultrasonic_thread = threading.Thread(target=ultrasonic_sensor)
ultrasonic_thread.daemon = True
ultrasonic_thread.start()

green_led.off()

# Dictionary to map sensor names to their audio messages
sensor_audio_messages = {
    "front_left": "Motion detected on front left",
    "front_right": "Motion detected on front right",
    "back_left": "Motion detected on back left",
    "back_right": "Motion detected on back right",
    "front": "Motion detected on front",
    "back": "Motion detected on back"
}

# Sharpening kernel (moderate sharpening)
sharp_kernel = np.array([[0, -0.5, 0],
                         [-0.5, 3, -0.5],
                         [0, -0.5, 0]])

# Edge enhancement kernel with less intensity
edge_kernel = np.array([[0, -0.25, 0],
                        [-0.25, 2, -0.25],
                        [0, -0.25, 0]])
                   

# Update display less frequently
display_interval = 0.1  # seconds
last_display_time = time.time()

# Adjusts the brightness and contrast
brightness = 20
contrast = 2.5 # Adjust contrast value as needed

# Function to increase color saturation
def increase_saturation(image, saturation_scale):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Split the HSV image into its channels
    h, s, v = cv2.split(hsv)
    
    # Increase saturation
    s = cv2.multiply(s, saturation_scale)
    
    # Clip the values to stay within the valid range [0, 255]
    s = np.clip(s, 0, 255).astype(np.uint8)
    
    # Merge the channels back
    hsv = cv2.merge([h, s, v])
    
    # Convert back to RGB color space
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return rgb

# Timing variables for object detection
last_detection_time = time.time()
detection_delay = 2  # seconds

# Detection control
detection = False
detection_voice = False
while True:
    # Check control pin state
    if control_pin.is_active:
        detection = True
        print("Object Detection Started")
    else:
        detection = False
        print("Object Detection Stopped")
    
    # Capture image from camera
    im = picam2.capture_array()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    # Apply brightness and contrast adjustments
    im = cv2.addWeighted(im, contrast, np.zeros(im.shape, im.dtype), 0, brightness)
    
    # Apply edge filter
    im = cv2.filter2D(im, -1, edge_kernel)
    
    # Increase color saturation
    im = increase_saturation(im, saturation_scale=1.5)
    
    # Perform object detection if detection is enabled and enough time has passed
    current_time = time.time()
    if detection:
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
        
        # Update the last detection time
        last_detection_time = current_time
    
    # Show the image with detection less frequently
    cv2.imshow('Image', im)
    cv2.waitKey(1)
    
    # Handle PIR sensors
    for sensor_name, sensor in pir_sensors.items():
        if sensor.motion_detected:
            print(f"Motion Detected by {sensor_name}")
            speak(sensor_audio_messages[sensor_name])
            
            # Wait for no motion and turn off LED
            sensor.wait_for_no_motion()
            green_led.off()
            print(f"Motion Stopped by {sensor_name}")
    
    # Sleep for a short time before the next loop iteration
    time.sleep(0.05)
