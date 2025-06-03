from django.shortcuts import render, redirect
from django.urls import reverse_lazy
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, authenticate, get_user_model
from django.contrib.messages.views import SuccessMessageMixin
from django.contrib.auth.views import LoginView
from django.contrib import messages
from django.core.mail import send_mail
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import cv2
import json
import base64
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import datetime

from .forms import CustomUserCreationForm, CustomAuthenticationForm
from ekyc import logger
from .models import CustomUser
from .HumanV import HumanVerificationSystem
from .NLP import UserInfo, get_user_info
from .OCR import extract_text_with_confidence, extract_aadhar_details, process_single_frame, process_multiple_frames

User = get_user_model()

@login_required
def ekyc_home(request):
    return render(request, 'home.html')

def signup_view(request):
    logger.info("into the signup page view")
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(email=user.email, password=raw_password)
            if user is not None:
                login(request, user)
                return redirect('ekyc_home')
    else:
        form = CustomUserCreationForm()
    return render(request, 'signup.html', {'form': form})

class Login(SuccessMessageMixin, LoginView):
    template_name = 'login.html'
    success_url = reverse_lazy('ekyc_home')
    success_message = "Successfully logged in!"
    form_class = CustomAuthenticationForm

    def get_success_url(self):
        return self.success_url

#--MAIN FEATURS HERE ---------------------------------
@login_required
def process_video(request):
    logger.info("INTO THE process_video PAGE RIGHT NOW!")
    if request.method == "POST":
        video_feed = request.FILES.get('video')
        liveness_score = 0.95
        return JsonResponse({'liveness_score': liveness_score})
    return JsonResponse({'error': 'invalide request'})


@login_required
def process_id_card(request):
    logger.info("INTO THE process_id_card PAGE RIGHT NOW!")
    if request.method == "POST":
        id_image = request.FILES.get('id_image')
        result = "Valid ID"
        return JsonResponse({'result': result})
    return JsonResponse({'error': 'invalide request'})

def home(request):
    return render(request, 'home.html')

@login_required
def human_verification(request):
    """Handle both GET (initial page load) and POST (frame processing) requests"""
    if request.method == 'GET':
        return render(request, 'human_verification.html')
    
    elif request.method == 'POST':
        try:
            data = json.loads(request.body)
            frame_data = data.get('frame')
            
            if not frame_data:
                return JsonResponse({
                    'status': 'error',
                    'message': 'No frame data provided'
                })

            # Process the frame using verifier
            frame_bytes = base64.b64decode(frame_data.split(',')[1])
            frame_arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)
            
            result = verifier.verify_frame(frame)
            return JsonResponse(result)

        except Exception as e:
            logger.error(f"Error in human verification: {str(e)}")
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            })
    
    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request method'
    })

@login_required
def nlp_process(request):
    try:
        stop_event = threading.Event()
        user_info = UserInfo()
        
        # Start NLP process
        info_thread = threading.Thread(target=get_user_info, args=(stop_event, user_info))
        info_thread.start()
        info_thread.join()

        # Check if all information was collected
        nlp_complete = not any(value.startswith("Waiting for") for value in [
            user_info.first_name, 
            user_info.last_name, 
            user_info.age, 
            user_info.phone
        ])

        return JsonResponse({
            'status': 'success',
            'nlp_complete': nlp_complete,
            'user_info': {
                'first_name': user_info.first_name,
                'last_name': user_info.last_name,
                'age': user_info.age,
                'phone': user_info.phone
            }
        })

    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        })

def get_current_prompt(request):
    from .NLP import current_prompt
    # logger.info(f"Current prompt being sent: {current_prompt}")
    print(current_prompt, "views script current_prompt")  
    return JsonResponse({'prompt': current_prompt, 'status': 'success'})

verifier = HumanVerificationSystem()

@csrf_exempt
def process_frame(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests are allowed'})

    try:
        data = json.loads(request.body)
        frame_data = data.get('frame')
        
        if not frame_data:
            return JsonResponse({'error': 'No frame data provided'})

        # Convert base64 to frame
        frame_bytes = base64.b64decode(frame_data.split(',')[1])
        frame_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)

        # Process frame
        result = verifier.verify_frame(frame)
        return JsonResponse(result)

    except Exception as e:
        return JsonResponse({'error': str(e)})

@csrf_exempt
def capture_photo(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests are allowed'})

    try:
        data = json.loads(request.body)
        frame_data = data.get('frame')
        
        if not frame_data:
            return JsonResponse({'error': 'No frame data provided'})

        # Convert base64 to frame
        frame_bytes = base64.b64decode(frame_data.split(',')[1])
        frame_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)

        # Save photo
        filename = verifier.save_photo(frame)
        if filename:
            return JsonResponse({'success': True, 'filename': filename})
        else:
            return JsonResponse({'error': 'Failed to save photo'})

    except Exception as e:
        return JsonResponse({'error': str(e)})

def get_most_frequent_data(all_results):
    """Get the most frequent value for each field from multiple OCR results"""
    field_values = {
        "name": [],
        "dob": [],
        "aadhar_number": [],
        "address": []
    }
    
    # Collect all values
    for result in all_results:
        for field in field_values.keys():
            if result.get(field):
                field_values[field].append(result[field])
    
    # Get most frequent value for each field
    final_result = {}
    for field, values in field_values.items():
        if values:
            # Use Counter to find most common value
            from collections import Counter
            counter = Counter(values)
            final_result[field] = counter.most_common(1)[0][0]
    
    return final_result

@csrf_exempt
def ocr_process(request):
    try:
        if request.method == 'POST':
            data = json.loads(request.body)
            frame_data = data.get('frame')
            elapsed_time = data.get('elapsed_time', 0)
            
            if not frame_data:
                return JsonResponse({
                    'status': 'error',
                    'message': 'No frame data provided'
                })

            # Process frame
            frame_bytes = base64.b64decode(frame_data.split(',')[1])
            frame_arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)

            if frame is None:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Invalid frame data'
                })

            # Store OCR results in session
            if not request.session.get('ocr_results'):
                request.session['ocr_results'] = []
            
            # Process the frame for OCR
            card_detected, result = process_multiple_frames(frame, num_frames=1)
            
            if card_detected and result:
                # Add new result to session
                request.session['ocr_results'].append(result)
                request.session.modified = True
                
                # If this is the final processing (-1) or we have good results
                if elapsed_time == -1 or (result.get('Name') and result.get('Aadhar Number')):
                    # Get most frequent values from all results
                    all_results = request.session['ocr_results']
                    final_result = get_most_frequent_data([{
                        'name': r.get('Name', ''),
                        'dob': r.get('DOB', ''),
                        'aadhar_number': r.get('Aadhar Number', ''),
                        'address': r.get('Address', '')
                    } for r in all_results])
                    
                    # Clear session data
                    request.session['ocr_results'] = []
                    
                    if final_result:
                        return JsonResponse({
                            'status': 'success',
                            'ocr_complete': True,
                            'ocr_data': final_result
                        })
            
            # If still processing
            if elapsed_time > 0:
                return JsonResponse({
                    'status': 'processing',
                    'message': 'Card detection in progress...'
                })
            
            # If timeout or final processing with no results
            return JsonResponse({
                'status': 'error',
                'message': 'Could not read card clearly'
            })

    except Exception as e:
        print(f"OCR Error: {str(e)}")
        return JsonResponse({
            'status': 'error',
            'message': 'Error processing card'
        })
