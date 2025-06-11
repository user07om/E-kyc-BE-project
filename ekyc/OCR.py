import cv2
import easyocr
import re
import os
import numpy as np
from PIL import Image
import time
import tempfile
import concurrent.futures
import uuid
from django.conf import settings

class OCRSystem:
    def __init__(self):
        self.aadhar_number = None
        self.dob = None
        self.name = None
        self.captured_card_path = None # To store the path of the saved image
        self.last_frame = None

    def save_card_image(self):
        """Saves the last processed frame to the media directory."""
        if self.captured_card_path or self.last_frame is None:
            return

        try:
            filename = f"{uuid.uuid4()}_card.jpg"
            filepath = os.path.join(settings.MEDIA_ROOT, filename)
            os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
            
            # The frame is a numpy array, save it with cv2
            cv2.imwrite(filepath, self.last_frame)
            
            self.captured_card_path = os.path.join(settings.MEDIA_URL, filename).replace("\\", "/")
            print(f"Saved card image to: {self.captured_card_path}")

        except Exception as e:
            print(f"Error saving card image: {e}")
            self.captured_card_path = None

def detect_aadhar_card(frame):
    """Detect if frame contains an Aadhar card"""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            rect = cv2.boundingRect(contour)
            x, y, w, h = rect
            area = w * h
            aspect_ratio = float(w)/h
            
            # Improved card detection criteria
            min_area = frame.shape[0] * frame.shape[1] // 6  # Card should be at least 1/6 of frame
            if 1.4 <= aspect_ratio <= 1.8 and area >= min_area:
                return True, rect
        
        return False, None
    except Exception as e:
        print(f"Error in detect_aadhar_card: {str(e)}")
        return False, None

def process_single_frame(frame):
    """Capture and process a single frame when Aadhar card is detected"""
    try:
        has_card, bounds = detect_aadhar_card(frame)
        if has_card:
            x, y, w, h = bounds
            card_image = frame[y:y+h, x:x+w]
            
            # Save the card image
            temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg', dir=temp_dir)
            image_path = temp_file.name
            temp_file.close()
            
            cv2.imwrite(image_path, card_image)
            print("Aadhar card detected and captured")
            
            # Process the image
            results = extract_text_with_confidence(image_path)
            if results:
                details = extract_aadhar_details(results)
                if details and details.get('Name'):
                    print("OCR successful!")
                    return True, details
                
            return True, None  # Card detected but OCR failed
        return False, None  # No card detected
            
    except Exception as e:
        print(f"Error in process_single_frame: {str(e)}")
        return False, None

def extract_text_with_confidence(image_path, min_confidence=0.4):
    """Extract text directly from image without enhancement"""
    try:
        print(f"Starting OCR on image: {image_path}")
        temp_dir = os.path.join(os.path.dirname(__file__), 'temp_ocr')
        os.makedirs(temp_dir, exist_ok=True)
        os.environ['EASYOCR_MODULE_PATH'] = temp_dir
        
        reader = easyocr.Reader(['en'], verbose=False)  # Using only English for faster processing
        
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return []
            
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image at {image_path}")
            return []
            
        # Direct OCR on original image
        print("Running OCR...")
        results = reader.readtext(image)
        
        # Filter results by confidence
        filtered_results = [(text, conf) for bbox, text, conf in results if conf > min_confidence]
        print(f"Found {len(filtered_results)} confident results")
        return filtered_results
        
    except Exception as e:
        print(f"OCR Error: {str(e)}")
        return []

def extract_aadhar_details(text_results):
    """Extract only essential Aadhar card details"""
    details = {
        "Name": "",
        "DOB": "",
        "Aadhar Number": "",
        "Confidence": {}
    }
    
    patterns = {
        "Aadhar Number": r'(?<!\d)(\d{4}\s\d{4}\s\d{4})(?!\d)',
        "DOB": r'(?:(?:DOB|Date of Birth|जन्म तारीख|जन्मतिथि)\s[:/]?\s)?(\d{1,2}[\/.-]\d{1,2}[\/.-]\d{4})',
        "Name": r'(?:Government\s+of\s+India|India).*?\b([A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?)\b(?=(?:DOB|Date\s+of\s+Birth|\d{1,2}[\/.-]\d{1,2}[\/.-]\d{4}))',
    }
    
    
    name_candidates = []
    
    for text, confidence in text_results:
        text = text.strip().upper()
        
        if len(text) < 3 or "GOVERNMENT OF INDIA" in text or "UNIQUE IDENTIFICATION" in text:
            continue
            
        # Check for Aadhar number
        if re.search(patterns["Aadhar Number"], text) and not details["Aadhar Number"]:
            aadhar_match = re.search(patterns["Aadhar Number"], text)
            details["Aadhar Number"] = aadhar_match.group(0)
            details["Confidence"]["Aadhar Number"] = confidence
            continue
            
        # Check for DOB
        if re.search(patterns["DOB"], text) and not details["DOB"]:
            dob_match = re.search(patterns["DOB"], text)
            details["DOB"] = dob_match.group(0)
            details["Confidence"]["DOB"] = confidence
            continue
            
        # Check for Name (text before DOB or "DOB")
        if re.search(patterns["Name"], text) and not details["Name"]:
            name_match = re.search(patterns["Name"], text)
            details["Name"] = name_match.group(1).strip()
            details["Confidence"]["Name"] = confidence
            continue
            
        # Collect potential name candidates
        if not details["Name"] and len(name_candidates) < 3 and len(text) > 3:
            name_candidates.append((text, confidence))
    
    # Process name if not already extracted
    if not details["Name"] and name_candidates:
        name_candidates.sort(key=lambda x: x[1], reverse=True)
        details["Name"] = name_candidates[0][0]
        details["Confidence"]["Name"] = name_candidates[0][1]
    
    print(details)
    return details


def compare_text_similarity(str1, str2):
    """Compare similarity between two strings"""
    if not str1 or not str2:
        return 0
    str1 = str1.strip().upper()
    str2 = str2.strip().upper()
    
    # For Aadhar number, remove spaces and compare
    if any(c.isdigit() for c in str1) and any(c.isdigit() for c in str2):
        str1 = ''.join(filter(str.isdigit, str1))
        str2 = ''.join(filter(str.isdigit, str2))
        return 1.0 if str1 == str2 else 0.0

    # For other text, compare character by character
    total_chars = max(len(str1), len(str2))
    if total_chars == 0:
        return 0
        
    matches = sum(1 for i in range(min(len(str1), len(str2))) if str1[i] == str2[i])
    return matches / total_chars

def get_most_consistent_result(all_results):
    """Get the most consistent result from multiple frames"""
    if not all_results:
        return None

    # Group results by field
    field_values = {
        "Name": [],
        "DOB": [],
        "Aadhar Number": []
    }

    # Collect all values for each field
    for result in all_results:
        if result:
            for field in field_values.keys():
                if result.get(field):
                    field_values[field].append(result[field])

    # Find most consistent value for each field
    best_result = {
        "Name": "",
        "DOB": "",
        "Aadhar Number": "",
        "Confidence": {}
    }

    for field, values in field_values.items():
        if not values:
            continue

        # Compare each value with all others to find the most consistent one
        best_score = 0
        best_value = ""
        
        for val1 in values:
            current_score = 0
            for val2 in values:
                similarity = compare_text_similarity(val1, val2)
                current_score += similarity
            
            if current_score > best_score:
                best_score = current_score
                best_value = val1

        if best_value:
            best_result[field] = best_value
            best_result["Confidence"][field] = best_score / len(values)  # Normalize confidence

    return best_result if any(best_result[k] for k in ["Name", "DOB", "Aadhar Number"]) else None

def process_multiple_frames(frame, num_frames=3):  # Reduced to 3 frames for faster processing
    """Process fewer frames for quicker results"""
    try:
        has_card, bounds = detect_aadhar_card(frame)
        if has_card:
            x, y, w, h = bounds
            card_image = frame[y:y+h, x:x+w]
            
            temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            
            # Process single frame first
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg', dir=temp_dir)
            image_path = temp_file.name
            temp_file.close()
            
            cv2.imwrite(image_path, card_image)
            results = extract_text_with_confidence(image_path)
            
            if results:
                details = extract_aadhar_details(results)
                if details and details.get('Name') and details.get('Aadhar Number'):
                    print("Got clear results from first frame")
                    os.remove(image_path)
                    return True, details
            
            os.remove(image_path)
            return True, None
            
        return False, None
        
    except Exception as e:
        print(f"Error in process_multiple_frames: {str(e)}")
        return False, None

def display_results(details):
    """Display extracted details in a formatted way"""
    print("\n" + "="*50)
    print(" AADHAR CARD INFORMATION EXTRACTION RESULTS ")
    print("="*50 + "\n")
    
    for field, value in details.items():
        if field != "Confidence":
            confidence = details["Confidence"].get(field, "N/A")
            confidence_str = f"{confidence:.2f}" if isinstance(confidence, float) else confidence
            if value:
                print(f"{field}: {value}")
                print(f"Confidence: {confidence_str}\n")
            
    print("="*50)
    print("Note: Lower confidence values indicate less reliable extraction")
    print("="*50)

def main():
    try:
        print("\nWelcome to Aadhar Card OCR System")
        print("----------------------------------")
        
        # Capture image with guide frame
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Add CAP_DSHOW for Windows
        if not cap.isOpened():
            raise Exception("Could not open camera")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                raise Exception("Could not read frame from camera")
            
            detected, details = process_multiple_frames(frame)
            if detected:
                if details:
                    print("Aadhar card details extracted successfully.")
                    display_results(details)
                else:
                    print("Aadhar card detected but OCR failed.")
                break
        
        cap.release()
        
        print("\nThank you for using Aadhar Card OCR System")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":  
    main()