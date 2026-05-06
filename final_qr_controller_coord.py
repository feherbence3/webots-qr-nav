import cv2
import numpy as np
import math
from controller import Robot

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# --- Camera Setup ---
cam = robot.getDevice('camera')
cam.enable(timestep)

qr_detector = cv2.QRCodeDetector()

# --- Motor Setup ---
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

cv2.namedWindow("What the Detector Sees", cv2.WINDOW_NORMAL)

# --- Constants & States ---
BASE_SPEED = 3.0
TURN_SPEED = 4.0      
TURN_TIME_90 = 0.70   

STATE_FORWARD = 0
STATE_TURNING = 1
STATE_WAITING = 2
STATE_STOPPED = 3

state = STATE_FORWARD
turn_dir = "right"
action_start_time = 0
action_cooldown_end = 0 

# --- NEW: Memory Variables ---
pending_command = None
frames_since_qr_lost = 0

def process_frame(frame):
    h, w = frame.shape[:2]
    
    # CROP: 40%
    crop_y = int(h * 0.40)
    roi = frame[crop_y:, :]
    roi_h, roi_w = roi.shape[:2]
    
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur_raw = cv2.GaussianBlur(gray_roi, (3, 3), 0)
    
    # --- 1. GENERATE AN ARSENAL OF SMART FILTERS ---
    
    # Filter A: Standard Adaptive (Good for even lighting)
    thresh_raw = cv2.adaptiveThreshold(
        blur_raw, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 7
    )
    
    # Filter B: LARGE Block Adaptive (MAGIC for thick bands of horizontal glare!)
    thresh_large = cv2.adaptiveThreshold(
        blur_raw, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 11
    )
    
    # Filter C: Morphological Close (Patches up broken QR blocks destroyed by noise)
    kernel = np.ones((3,3), np.uint8)
    thresh_morph = cv2.morphologyEx(thresh_raw, cv2.MORPH_CLOSE, kernel)
    
    # Filter D: CLAHE (Balances global contrast)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray_clahe = clahe.apply(gray_roi)
    thresh_clahe = cv2.adaptiveThreshold(
        cv2.GaussianBlur(gray_clahe, (3, 3), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 7
    )
    
    # --- 2. ROBUST QR DETECTION ---
    qr_command = None
    qr_bbox = None
    best_area = 0
    
    # Throw the smart filters at the detector. 
    candidates = [roi, gray_roi, thresh_raw, thresh_large, thresh_morph, gray_clahe, thresh_clahe]
    
    for img_candidate in candidates:
        data, bbox, _ = qr_detector.detectAndDecode(img_candidate)
        
        if bbox is not None:
            # Keep track of the largest/cleanest bounding box for the yellow dot tracking, 
            # even if we haven't successfully read the text yet.
            area = cv2.contourArea(np.int32(bbox))
            if area > best_area:
                best_area = area
                qr_bbox = bbox
                
        if data:
            # We successfully read the text! Lock it in and stop checking.
            qr_command = data.strip().upper()
            qr_bbox = bbox # Use the exact box that successfully read the data
            break 
            
    qr_bottom_y = None
    
    # --- 3. LINE THRESHOLDING & MASKING ---
    # Use CLAHE for the line follower so it ignores glare on the floor
    _, thresh_lines = cv2.threshold(gray_clahe, 80, 255, cv2.THRESH_BINARY_INV)
    display_img = cv2.cvtColor(thresh_lines, cv2.COLOR_GRAY2BGR)
    
    if qr_bbox is not None:
        bbox_int = np.int32(qr_bbox).reshape(-1, 1, 2)
        cv2.fillPoly(thresh_lines, [bbox_int], 0) 
        cv2.polylines(display_img, [bbox_int], True, (255, 0, 255), 3) 
        
        pts = qr_bbox[0]
        qr_bottom_y = np.max(pts[:, 1])
        lowest_pt_idx = np.argmax(pts[:, 1])
        
        cv2.circle(display_img, (int(pts[lowest_pt_idx, 0]), int(qr_bottom_y)), 6, (0, 255, 255), -1)
        
        if qr_command:
            cv2.putText(display_img, qr_command, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        else:
            # Shows that it found the shape, but is still fighting glare to read it
            cv2.putText(display_img, "???", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    TRIGGER_Y = int(roi_h * 0.85) 
    cv2.line(display_img, (0, TRIGGER_Y), (roi_w, TRIGGER_Y), (0, 0, 255), 2)
    cv2.putText(display_img, "TRIGGER LINE", (10, TRIGGER_Y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # --- 4. FIND CORRIDOR LINES & HORIZONTAL GRID LINES ---
    lines = cv2.HoughLinesP(thresh_lines, rho=1, theta=np.pi/180, threshold=40, 
                            minLineLength=25, maxLineGap=10)
    
    left_candidates = []
    right_candidates = []
    horizontal_y = None 
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(display_img, (x1, y1), (x2, y2), (0, 70, 0), 1)
            
            if y1 < y2:
                x1, y1, x2, y2 = x2, y2, x1, y1
                
            dx = x2 - x1
            dy = y1 - y2
            
            if dy == 0: 
                angle = 90
            else:
                angle = math.degrees(math.atan2(dx, dy)) 
            
            if abs(angle) < 35:
                x_bottom = x1 + (dx / dy) * (y1 - roi_h)
                if x_bottom < roi_w / 2:
                    left_candidates.append((x_bottom, angle, x1, y1, x2, y2))
                else:
                    right_candidates.append((x_bottom, angle, x1, y1, x2, y2))
            
            elif abs(angle) > 50:
                cv2.line(display_img, (x1, y1), (x2, y2), (255, 255, 0), 2) 
                avg_y = (y1 + y2) / 2
                if horizontal_y is None or avg_y > horizontal_y:
                    horizontal_y = avg_y

    best_left = max(left_candidates, key=lambda x: x[0]) if left_candidates else None
    best_right = min(right_candidates, key=lambda x: x[0]) if right_candidates else None

    CORRIDOR_WIDTH = int(roi_w * 0.8) 
    midpoint_x = None
    best_vertical_angle = None

    if best_left and best_right:
        midpoint_x = (best_left[0] + best_right[0]) / 2
        best_vertical_angle = (best_left[1] + best_right[1]) / 2
        cv2.line(display_img, (best_left[2], best_left[3]), (best_left[4], best_left[5]), (0, 255, 0), 2)
        cv2.line(display_img, (best_right[2], best_right[3]), (best_right[4], best_right[5]), (0, 255, 0), 2)
        
    elif best_left:
        midpoint_x = best_left[0] + (CORRIDOR_WIDTH / 2)
        best_vertical_angle = best_left[1]
        cv2.line(display_img, (best_left[2], best_left[3]), (best_left[4], best_left[5]), (0, 200, 255), 2)
        
    elif best_right:
        midpoint_x = best_right[0] - (CORRIDOR_WIDTH / 2)
        best_vertical_angle = best_right[1]
        cv2.line(display_img, (best_right[2], best_right[3]), (best_right[4], best_right[5]), (0, 200, 255), 2)
        
    if midpoint_x is not None:
        best_center_error = midpoint_x - (roi_w / 2)
        cv2.circle(display_img, (int(midpoint_x), roi_h - 2), 4, (0, 0, 255), -1)
    else:
        best_center_error = None

    cv2.line(display_img, (roi_w//2, 0), (roi_w//2, roi_h), (255, 0, 0), 1)
                    
    return best_center_error, best_vertical_angle, display_img, qr_command, qr_bottom_y, horizontal_y

# ────────────────────────────────────────────────
# Main Loop
# ────────────────────────────────────────────────
current_x, current_y = 0, 1
heading_x, heading_y = 0, 1
last_grid_cross_time = 0
is_on_intersection = False 

# --- Smart Ignore Variables ---
last_completed_action = None
ignore_action_until = 0

print("Starting Precision QR Navigation with Grid Mapper...")

while robot.step(timestep) != -1:
    raw_image = cam.getImage()
    
    if raw_image is not None:
        frame = np.frombuffer(raw_image, np.uint8).reshape((cam.getHeight(), cam.getWidth(), 4))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        error_x, angle, debug_view, current_command, qr_bottom_y, horizontal_y = process_frame(frame)
        
        cv2.imshow("What the Detector Sees", debug_view)
        cv2.waitKey(1)
        
        current_time = robot.getTime()
        
        # --- State Machine ---
        
        if state == STATE_FORWARD:
            # --- NEW: Decoupled Triggers ---
            # QR codes trigger at 85% down the screen (perfect for turning)
            action_trigger_y = (cam.getHeight() * 0.60) * 0.85 
            # Grid lines are counted at 70% down the screen (prevents missing them if the frame skips!)
            grid_trigger_y = (cam.getHeight() * 0.60) * 0.70 
            
            # --- 0. GRID TRACKING ---
            is_crossing_now = False
            if qr_bottom_y is not None and qr_bottom_y >= grid_trigger_y:
                is_crossing_now = True 
            elif horizontal_y is not None and horizontal_y >= grid_trigger_y:
                is_crossing_now = True
                
            if is_crossing_now:
                if not is_on_intersection and (current_time - last_grid_cross_time > 0.4):
                    current_x += heading_x
                    current_y += heading_y
                    last_grid_cross_time = current_time
                    is_on_intersection = True
            else:
                is_on_intersection = False 
            
            # 1. SMART MEMORY
            if current_command and current_time > action_cooldown_end:
                if current_command == last_completed_action and current_time < ignore_action_until:
                    pass 
                elif pending_command != current_command:
                    pending_command = current_command
                    print(f"[{current_time:.1f}] Saw '{pending_command}' ahead. Waiting to trigger...")
            
            # 2. EXECUTION TRIGGER
            if pending_command:
                execute_now = False
                
                if qr_bottom_y is not None:
                    frames_since_qr_lost = 0
                    # Use the action_trigger for QR codes
                    if qr_bottom_y >= action_trigger_y:
                        execute_now = True
                else:
                    frames_since_qr_lost += 1
                    if frames_since_qr_lost > 6:
                        execute_now = True

                if execute_now:
                    # --- Calculate exact QR Map Position ---
                    qr_x = current_x + heading_x
                    qr_y = current_y + heading_y
                    
                    print(f"\n================================================")
                    print(f"=== EXECUTING: {pending_command}")
                    print(f"=== QR CODE LOCATED AT: [{qr_x}, {qr_y}]")
                    print(f"================================================\n")
                    
                    if pending_command == "RIGHT":
                        state = STATE_TURNING
                        turn_dir = "right"
                        action_start_time = current_time
                        heading_x, heading_y = heading_y, -heading_x 
                        
                    elif pending_command == "LEFT":
                        state = STATE_TURNING
                        turn_dir = "left"
                        action_start_time = current_time
                        heading_x, heading_y = -heading_y, heading_x
                        
                    elif pending_command == "WAIT_5S":
                        state = STATE_WAITING
                        action_start_time = current_time
                        left_motor.setVelocity(0.0)
                        right_motor.setVelocity(0.0)
                        
                    elif pending_command == "STOP":
                        state = STATE_STOPPED
                        left_motor.setVelocity(0.0)
                        right_motor.setVelocity(0.0)
                        
                    pending_command = None 
                    frames_since_qr_lost = 0

            # 3. NORMAL DRIVING
            if state == STATE_FORWARD: 
                if error_x is not None and angle is not None:
                    k_p_offset = 0.02
                    k_p_angle = 0.04
                    correction = (error_x * k_p_offset) + (angle * k_p_angle)
                    
                    left_v = max(-6.28, min(BASE_SPEED + correction, 6.28))
                    right_v = max(-6.28, min(BASE_SPEED - correction, 6.28))
                    
                    left_motor.setVelocity(left_v)
                    right_motor.setVelocity(right_v)
                else:
                    left_motor.setVelocity(BASE_SPEED)
                    right_motor.setVelocity(BASE_SPEED)
                    
        elif state == STATE_TURNING:
            if turn_dir == "right":
                left_motor.setVelocity(TURN_SPEED)
                right_motor.setVelocity(-TURN_SPEED)
            else:
                left_motor.setVelocity(-TURN_SPEED)
                right_motor.setVelocity(TURN_SPEED)
                
            if current_time >= action_start_time + TURN_TIME_90:
                print("Turn complete.")
                state = STATE_FORWARD
                action_cooldown_end = current_time + 0.5 
                last_completed_action = turn_dir.upper() 
                ignore_action_until = current_time + 2.0 
                last_grid_cross_time = current_time 
                
                current_x += heading_x
                current_y += heading_y
                
        elif state == STATE_WAITING:
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)
            
            if current_time >= action_start_time + 5.0:
                print("Wait complete. Resuming...")
                state = STATE_FORWARD
                action_cooldown_end = current_time + 0.2 
                last_completed_action = "WAIT_5S"        
                ignore_action_until = current_time + 3.0 
                last_grid_cross_time = current_time 
                
                current_x += heading_x
                current_y += heading_y
                
        elif state == STATE_STOPPED:
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)
                
        elif state == STATE_STOPPED:
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)