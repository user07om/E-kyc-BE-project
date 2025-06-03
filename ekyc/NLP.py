import cv2
import speech_recognition as sr
import threading
import time
import re

# Global variable to store current prompt
current_prompt = None

class UserInfo:
    def __init__(self):
        self.first_name = "Waiting for first name..."
        self.last_name = "Waiting for last name..."
        self.age = "Waiting for age..."
        self.phone = "Waiting for phone number..."

def process_camera(stop_event, user_info):
    # Initialize camera before other operations to open webcam ASAP
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        stop_event.set()
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (255, 255, 255)
    thickness = 2

    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            # Add semi-transparent background for better text visibility
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 30), (400, 130), (0, 0, 0), -1)
            alpha = 0.7  # Transparency factor
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # Display user information
            cv2.putText(frame, f"First Name: {user_info.first_name}", (20, 50), font, 
                        font_scale, font_color, thickness)
            cv2.putText(frame, f"Last Name: {user_info.last_name}", (20, 75), font, 
                        font_scale, font_color, thickness)
            cv2.putText(frame, f"Age: {user_info.age}", (20, 100), font, 
                        font_scale, font_color, thickness)
            cv2.putText(frame, f"Phone: {user_info.phone}", (20, 125), font, 
                        font_scale, font_color, thickness)
            
            cv2.imshow('User Information', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
        else:
            print("Error: Failed to capture frame from webcam")
            time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()

# def get_speech_input(prompt, validation_func=None, error_message=None, max_attempts=5):
#     recognizer = sr.Recognizer()
#     attempt = 0
    
#     while attempt < max_attempts:
#         try:
#             with sr.Microphone() as source:
#                 print(f"\n{prompt}")
#                 # Shorter ambient noise adjustment for faster response
#                 recognizer.adjust_for_ambient_noise(source, duration=0.3)
#                 print("Listening...")
                
#                 # Increased timeout and phrase_time_limit for better recognition
#                 audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)
#                 text = recognizer.recognize_google(audio)
#                 text = text.strip()
                
#                 # Apply validation if provided
#                 if validation_func and not validation_func(text):
#                     print(error_message)
#                     attempt += 1
#                     continue
                
#                 return text
                
#         except sr.UnknownValueError:
#             print("Sorry, I couldn't understand that. Please speak clearly.")
#         except sr.RequestError:
#             print("Speech recognition service unavailable. Please try again.")
#         except Exception as e:
#             print(f"Error: {e}. Please try again.")
        
#         attempt += 1
    
#     print(f"Maximum attempts reached. Using default value.")
#     return None


def get_speech_input(prompt, validation_func=None, error_message=None, max_attempts=5):
    global current_prompt
    recognizer = sr.Recognizer()
    attempt = 0
    
    # Update the current prompt
    current_prompt = prompt
    
    while attempt < max_attempts:
        try:
            with sr.Microphone() as source:
                print(f"\n{prompt}")
                recognizer.adjust_for_ambient_noise(source, duration=0.3)
                print("Listening...")
                
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)
                text = recognizer.recognize_google(audio)
                text = text.strip()
                
                if validation_func and not validation_func(text):
                    current_prompt = f"{error_message}\n{prompt}"
                    print(error_message)
                    attempt += 1
                    continue

                # current_prompt = text
                # print(current_prompt, "inside the get_speech_input text getting!")
                return text
                
        except sr.UnknownValueError:
            current_prompt = "Please speak clearly.\n" + prompt
            print(current_prompt)
        except sr.RequestError:
            current_prompt = "Please try again.\n" + prompt
            print(current_prompt)
        except Exception as e:
            current_prompt = "Please try again.\n" + prompt
            print(current_prompt)
        
        attempt += 1
    
    # current_prompt = "Maximum attempts reached. Using default value."
    # print(current_prompt)
    return None


def validate_age(age_text):
    try:
        # Try to handle both text and numeric inputs
        age_text = age_text.lower().replace("years", "").replace("year", "").strip()
        age = int(age_text)
        return age >= 18
    except ValueError:
        return False

def validate_phone(phone_text):
    # Remove common speech recognition artifacts
    phone_text = phone_text.replace(" ", "").replace("-", "")
    phone_pattern = re.compile(r'^\d{10}$')
    return bool(phone_pattern.match(phone_text))

def format_phone(phone_text):
    # Format phone number nicely for display
    phone_text = phone_text.replace(" ", "").replace("-", "")
    if len(phone_text) == 10:
        return f"{phone_text[:3]}-{phone_text[3:6]}-{phone_text[6:]}"
    return phone_text

def get_user_info(stop_event, user_info):
    global current_prompt
    try:
        # Get first name
        first_name = get_speech_input("Please say your first name:")
        if first_name:
            user_info.first_name = first_name.capitalize()
            print(f"Recorded first name: {user_info.first_name}")
            current_prompt = f"Recorded First Name: {user_info.first_name}"
        else:
            user_info.first_name = "John"  # Dummy data
            current_prompt = "Using default name: John"
        
        # Get last name
        last_name = get_speech_input("Please say your last name:")
        if last_name:
            user_info.last_name = last_name.capitalize()
            print(f"Recorded last name: {user_info.last_name}")
            current_prompt = f"Recorded Last Name: {user_info.last_name}"
        else:
            user_info.last_name = "Doe"  # Dummy data
            current_prompt = "Using default last name: Doe"
        
        # Get age with validation
        age = get_speech_input(
            "Please say your age (must be 18 or older):",
            validate_age,
            "Age must be 18 or older. Please try again."
        )
        if age:
            age_value = int(age.lower().replace("years", "").replace("year", "").strip())
            user_info.age = f"{age_value} years"
            print(f"Recorded age: {user_info.age}")
            current_prompt = f"Recorded Age: {user_info.age}"
        else:
            user_info.age = "25 years"  # Dummy data
            current_prompt = "Using default age: 25 years"
        
        # Get phone number with validation
        phone = get_speech_input(
            "Please say your 10-digit mobile number (e.g., 1234567890):",
            validate_phone,
            "Please provide a valid 10-digit mobile number."
        )
        if phone:
            user_info.phone = format_phone(phone)
            print(f"Recorded phone: {user_info.phone}")
            current_prompt = f"Recorded Phone: {user_info.phone}"
        else:
            user_info.phone = "123-456-7890"  # Dummy data
            current_prompt = "Using default phone: 123-456-7890"
        
        print("\nAll information collected. Press 'q' to quit...")
        current_prompt = None
        
    except Exception as e:
        print(f"An error occurred during information collection: {e}")
        # Set dummy data for all fields if there's an error
        user_info.first_name = "John"
        user_info.last_name = "Doe"
        user_info.age = "25 years"
        user_info.phone = "123-456-7890"
        current_prompt = "Using default information due to error"
    finally:
        if all(value == "Waiting for name..." for value in [user_info.first_name, user_info.last_name, user_info.age, user_info.phone]):
            print("No information was collected successfully.")
            stop_event.set()

def main():
    stop_event = threading.Event()
    user_info = UserInfo()
    
    # Start camera thread immediately
    camera_thread = threading.Thread(target=process_camera, args=(stop_event, user_info))
    camera_thread.daemon = True  # Make thread daemon so it exits when main thread exits
    camera_thread.start()
    
    # Give the camera thread a moment to initialize
    time.sleep(0.5)
    
    # Start user info collection
    info_thread = threading.Thread(target=get_user_info, args=(stop_event, user_info))
    info_thread.start()

    try:
        info_thread.join()
        print("Information collection complete. Starting OCR process...")
        
        # Start OCR process
        from .OCR import main as ocr_main
        ocr_main()  # Execute OCR process
        
        print("OCR process complete. Press 'q' in the webcam window to exit.")
        
        while not stop_event.is_set() and camera_thread.is_alive():
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    finally:
        stop_event.set()
        time.sleep(0.5)  # Give threads time to clean up
        print("\nProgram terminated")

if __name__ == "__main__":
    main()
