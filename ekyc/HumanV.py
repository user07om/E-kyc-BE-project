import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
import os
import base64
import uuid
from django.conf import settings

class HumanVerificationSystem:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            refine_landmarks=True
        )
        self.blink_counter = 0
        self.blink_timestamps = []
        self.eye_heights = deque(maxlen=5)
        self.baseline_height = None
        self.blinks_verified = False
        self.current_phase = "blink"
        self.showing_excellent = False
        self.excellent_start_time = None
        self.captured_image_path = None
        self.current_frame = None

    def get_eye_height(self, landmarks, top_idx, bottom_idx):
        top = landmarks[top_idx]
        bottom = landmarks[bottom_idx]
        return np.sqrt((top.x - bottom.x)**2 + (top.y - bottom.y)**2)

    def detect_blink(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return

        landmarks = results.multi_face_landmarks[0].landmark
        
        left_height = self.get_eye_height(landmarks, 386, 374)
        right_height = self.get_eye_height(landmarks, 159, 145)
        current_height = (left_height + right_height) / 2
        
        if self.baseline_height is None and len(self.eye_heights) == 5:
            self.baseline_height = np.mean(self.eye_heights)
            
        self.eye_heights.append(current_height)

        if self.baseline_height is not None:
            height_ratio = current_height / self.baseline_height

            if len(self.blink_timestamps) == 0 or \
               (time.time() - self.blink_timestamps[-1] >= 0.4):
                
                if height_ratio < 0.6:
                    if len(self.blink_timestamps) == 0 or \
                       time.time() - self.blink_timestamps[-1] >= 0.4:
                        self.blink_counter += 1
                        self.blink_timestamps.append(time.time())
                        
                        if self.blink_counter >= 3:
                            self.blinks_verified = True

    def save_capture(self, frame):
        """Saves the captured frame to the media directory."""
        if self.captured_image_path:
            return # Already saved

        try:
            filename = f"{uuid.uuid4()}.jpg"
            # Note: MEDIA_ROOT is already an absolute path
            filepath = os.path.join(settings.MEDIA_ROOT, filename)
            
            # Ensure the directory exists
            os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
            
            cv2.imwrite(filepath, frame)
            
            # Store the URL path, not the filesystem path
            self.captured_image_path = os.path.join(settings.MEDIA_URL, filename).replace("\\", "/")
            print(f"Saved human verification image to: {self.captured_image_path}")

        except Exception as e:
            print(f"Error saving human verification image: {e}")
            self.captured_image_path = None

    def save_photo(self, frame):
        try:
            if not os.path.exists('captured_photos'):
                os.makedirs('captured_photos')
            
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f'captured_photos/photo_{timestamp}.jpg'
            cv2.imwrite(filename, frame)
            return filename
        except Exception as e:
            print(f"Error saving photo: {str(e)}")
            return None

    def get_frame_data(self):
        """Convert current frame to base64 for web display"""
        if self.current_frame is None:
            return None
        
        _, buffer = cv2.imencode('.jpg', self.current_frame)
        frame_data = base64.b64encode(buffer).decode('utf-8')
        return frame_data

    def process_frame(self, frame):
        """Process a single frame and return verification status"""
        self.current_frame = frame
        status = {
            'blink_count': self.blink_counter,
            'phase_complete': False,
            'message': '',
            'success': False
        }

        if self.showing_excellent:
            if time.time() - self.excellent_start_time <= 3.0:
                status['message'] = "Excellent!"
            else:
                self.showing_excellent = False
        else:
            self.detect_blink(frame)
            status['message'] = f"Blinks detected: {self.blink_counter}/3"

        if self.blinks_verified:
            status['phase_complete'] = True
            status['success'] = True
            status['message'] = "Verification complete!"
            if not self.captured_image_path:
                self.save_capture(self.current_frame)
            status['captured_image_path'] = self.captured_image_path

        return status

    def verify_frame(self, frame):
        """Single frame verification for Django integration"""
        if frame is None:
            return {
                'status': 'error',
                'message': 'No frame provided'
            }

        try:
            status = self.process_frame(frame)
            frame_data = self.get_frame_data()
            
            return {
                'status': 'success',
                'blink_count': self.blink_counter,
                'phase_complete': self.blinks_verified,
                'message': status['message'],
                'frame_data': frame_data,
                'captured_image_path': status.get('captured_image_path')
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }