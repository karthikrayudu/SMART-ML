import cv2
import numpy as np
import speech_recognition as sr
import pyttsx3
import threading
import time
import tkinter as tk
from PIL import Image, ImageTk

# Global variable to signal the end of the interview
interview_ended = False

# Function to capture video from webcam
def capture_video():
    global interview_ended
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        interview_ended = True
        return

    while not interview_ended:
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            frame = Image.fromarray(frame)  # Convert to Image
            frame = ImageTk.PhotoImage(frame)  # Convert to PhotoImage
            panel1.config(image=frame)
            panel1.image = frame

    cap.release()
    cv2.destroyAllWindows()

# Function to capture audio
def capture_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please answer the question...")
        audio = r.listen(source)

    try:
        response = r.recognize_google(audio)
        print("You answered: " + response)
        return response
    except sr.UnknownValueError:
        print("Speech recognition could not understand audio")
        return ""
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return ""

# Function to speak questions using text-to-speech
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Adjust the speaking rate (words per minute)
    engine.say(text)
    engine.runAndWait()

# Function to ask questions and capture responses
def conduct_interview(questions):
    global interview_ended
    for question in questions:
        if interview_ended:
            break
        speak("Question: " + question)
        time.sleep(1)  # Wait for the speaking to start
        answer = capture_audio()
        print("Interviewee's answer:", answer)

# Main function
def main():
    # Create the main GUI window
    root = tk.Tk()
    root.title("Interview Panel")

    # Create two panels: one for video and one for text
    panel1 = tk.Label(root)
    panel1.pack(side="left", padx=10, pady=10)

    panel2 = tk.Label(root, text="AI asking questions")
    panel2.pack(side="right", padx=10, pady=10)

    # Example interview questions
    questions = [
        "Tell me about yourself.",
        "What are your strengths and weaknesses?",
        "Why do you want to work for our company?",
        "Can you describe a challenging situation you faced at work and how you handled it?",
        # Add more questions as needed
    ]

    # Create thread for video capture
    video_thread = threading.Thread(target=capture_video)

    # Start the video capture thread
    video_thread.start()

    # Wait for a short delay to ensure the video stream has started
    time.sleep(2)

    # Conduct the interview
    conduct_interview(questions)

    # Signal the end of the interview to the video thread
    global interview_ended
    interview_ended = True

    # Wait for the video thread to finish
    video_thread.join()

    root.mainloop()

if __name__ == "__main__":
    main()
