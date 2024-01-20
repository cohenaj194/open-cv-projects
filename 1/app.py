import cv2
import numpy as np

# Flag to indicate if the application should be closed
#exit_flag = False

# Function to handle keyboard events
#def keyboard_event_handler(event):
#    global exit_flag

#    if event == ord('e'):
#        exit_flag = True

# Load the pre-trained Haar Cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load an image
image = cv2.imread(r'C:\Users\MSTAM\OneDrive\Documents\GitHub\open-cv-projects\1\Human_faces.jpg')  # Replace with your image path

if image is None:
    raise ValueError("Image not found or cannot be loaded.")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    # Draw a rectangle around the face
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    
    # Detect eyes within the face region
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (eye_x, eye_y, eye_w, eye_h) in eyes:
        # Draw a rectangle around the eyes
        cv2.rectangle(roi_color, (eye_x, eye_y), (eye_x+eye_w, eye_y+eye_h), (0, 255, 0), 2)

# Display the result
cv2.namedWindow('Facial Feature Art', cv2.WINDOW_NORMAL)
cv2.imshow('Facial Feature Art', image)
# Set the keyboard event callback function
#cv2.setMouseCallback('Facial Feature Art', keyboard_event_handler)

#while not exit_flag:
    # Display the image
#    cv2.imshow('Facial Feature Art', image)

#    key = cv2.waitKey(1)
#    if key == ord('f') and cv2.getWindowProperty('Facial Feature Art', cv2.WND_PROP_VISIBLE):
        # Create a file menu
#        menu = np.zeros((50, 200, 3), np.uint8)
#        menu[:] = (255, 255, 255)

        # Display "Close" option in the file menu
#        cv2.putText(menu, 'Close (Alt + e)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

#        cv2.imshow('File Menu', menu)
#    elif key == 27 or key == ord('e'):
        # Close the file menu
#        cv2.destroyWindow('File Menu')
cv2.waitKey(0)
cv2.destroyAllWindows()
