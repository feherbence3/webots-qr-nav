import cv2
import numpy as np
import math
from controller import Robot

robot = Robot()
timestep = int(robot.getBasicTimeStep())

cam = robot.getDevice('camera')
cam.enable(timestep)

detector = cv2.QRCodeDetector()

left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# 1. Enable Position Sensors (Encoders)
left_sensor = robot.getDevice('left wheel sensor')
right_sensor = robot.getDevice('right wheel sensor')
left_sensor.enable(timestep)
right_sensor.enable(timestep)

#cv2.namedWindow("What the Detector Sees", cv2.WINDOW_NORMAL)

# ────────────────────────────────────────────────
# Turn parameters
MAX_SPEED = 6.28
TURN_SPEED = 1.5  # <-- Lowered to prevent skidding/inertia overshoot

# 2. E-puck physical dimensions (in meters)
WHEEL_RADIUS = 0.0205
AXLE_LENGTH = 0.0528  # Distance between wheels

# 3. Calibration Tweak (1.0 is default math)
# Since you overshot, we lower this to tell the wheels to spin slightly less.
TURN_TWEAK = 0.95 

# Calculate exactly how many radians the wheel needs to spin
TARGET_WHEEL_RADS = ((AXLE_LENGTH / 2.0) * (math.pi / 2.0) / WHEEL_RADIUS) * TURN_TWEAK

def turn_90_left():
    print(f"Turning LEFT 90° (Target Rads: {TARGET_WHEEL_RADS:.2f})")
    start_pos = right_sensor.getValue()
    
    left_motor.setVelocity(-TURN_SPEED)
    right_motor.setVelocity(TURN_SPEED)
    
    # Spin until the right wheel has traveled the exact mathematical distance
    while robot.step(timestep) != -1:
        current_pos = right_sensor.getValue()
        if abs(current_pos - start_pos) >= TARGET_WHEEL_RADS:
            break
            
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)
    robot.step(timestep) 

def turn_90_right():
    print(f"Turning RIGHT 90° (Target Rads: {TARGET_WHEEL_RADS:.2f})")
    start_pos = left_sensor.getValue()
    
    left_motor.setVelocity(TURN_SPEED)
    right_motor.setVelocity(-TURN_SPEED)
    
    # Spin until the left wheel has traveled the exact mathematical distance
    while robot.step(timestep) != -1:
        current_pos = left_sensor.getValue()
        if abs(current_pos - start_pos) >= TARGET_WHEEL_RADS:
            break
            
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)
    robot.step(timestep)

# ────────────────────────────────────────────────
while robot.step(timestep) != -1:
    raw_image = cam.getImage()
    
    if raw_image is not None:
        frame = np.frombuffer(raw_image, np.uint8).reshape((cam.getHeight(), cam.getWidth(), 4))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        height, width = frame.shape[:2]
        start_row = int(height * 0.40)
        cropped = frame[start_row:, :]
        
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            25, 7
        )
        
        candidates = [
            ("gray ", gray),
            ("thresh", thresh),
            ("cropped color", cropped)
        ]
        
        detected = False
        data = ""           
        bbox = None
        
        for name, img in candidates:
            dec_data, dec_bbox, _ = detector.detectAndDecode(img)
            
            if dec_data:
                data = dec_data
                bbox = dec_bbox
                print(f"!!! DECODED from {name}: {data} !!!")
                detected = True
                
                if bbox is not None:
                    bbox = np.int32(bbox).reshape(-1, 1, 2)
                    cv2.polylines(cropped, [bbox], True, (0, 255, 0), 2)
                break
                
            elif dec_bbox is not None:
                print(f"→ Detected QR corners on {name}, but decode failed")
                bbox = dec_bbox
                bbox = np.int32(bbox).reshape(-1, 1, 2)
                cv2.polylines(cropped, [bbox], True, (0, 0, 255), 2)
                detected = True
        
        if not detected:
            print("No QR found this frame")
        
        display_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        cv2.addWeighted(display_img, 0.4, cropped, 0.6, 0, display_img)
        
        cv2.imshow("What the Detector Sees", display_img)
        cv2.waitKey(1)
    
    # ────────────────────────────────────────────────
    # Movement logic
    if detected and data:
        command = data.lower().strip()
        
        if "left" in command:
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)
            robot.step(timestep)   
            turn_90_left()
            
            # Briefly force it forward so it doesn't double-read the QR code
            left_motor.setVelocity(3.0)
            right_motor.setVelocity(3.0)
            end_forward = robot.getTime() + 0.5
            while robot.getTime() < end_forward:
                robot.step(timestep)
        
        elif "right" in command:
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)
            robot.step(timestep)
            turn_90_right()
            
            # Briefly force it forward so it doesn't double-read the QR code
            left_motor.setVelocity(3.0)
            right_motor.setVelocity(3.0)
            end_forward = robot.getTime() + 0.5
            while robot.getTime() < end_forward:
                robot.step(timestep)
        
        elif "stop" in command:
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)
        
        else:
            left_motor.setVelocity(3.0)
            right_motor.setVelocity(3.0)
    
    else:
        left_motor.setVelocity(1.5)
        right_motor.setVelocity(1.5)