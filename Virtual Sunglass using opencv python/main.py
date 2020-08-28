import cv2
import numpy as np
import dlib
from math import hypot

cap = cv2.VideoCapture(0)
eye_image = cv2.imread("Sunglass.png")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
def action():
    try:
        while True:
            _, frame = cap.read()
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = detector(frame)
    
            for face in faces:
                landmarks = predictor(gray_frame, face)

                top_eye = (landmarks.part(27).x, landmarks.part(27).y)
                left_eye = (landmarks.part(36).x, landmarks.part(36).y)
                right_eye = (landmarks.part(45).x, landmarks.part(45).y)
                center_eye = (landmarks.part(28).x, landmarks.part(28).y)
            
                eye_width = int(hypot(left_eye[0] - right_eye[0], left_eye[1] - right_eye[1])*2.5)
                eye_height = int(eye_width * 0.5633)
    
                top_left = (int(center_eye[0] - eye_width/2), int(center_eye[1] - eye_height/2))
                bottom_right = (int(center_eye[0] + eye_width/2), int(center_eye[1] + eye_height/2))
            
                #cv2.rectangle(frame, (int(center_eye[0] - eye_width/2), int(center_eye[1] - eye_height/2)), (int(center_eye[0] + eye_width/2), int(center_eye[1] + eye_height/2)), (0, 255, 0),2)
            
                eye_sunglass = cv2.resize(eye_image,(eye_width, eye_height))
                eye_sunglass_gray = cv2.cvtColor(eye_sunglass, cv2.COLOR_BGR2GRAY)
                
                _, eye_mask = cv2.threshold(eye_sunglass_gray, 0, 255, cv2.THRESH_BINARY_INV)
        
                eye_area = frame[top_left[1] : top_left[1] + eye_height, top_left[0] : top_left[0] + eye_width]
        
                eye_area_no_eye = cv2.bitwise_and(eye_area, eye_area, mask = eye_mask)
            
                final_eye = cv2.add(eye_area_no_eye, eye_sunglass)
    
                frame[top_left[1] : top_left[1] + eye_height, top_left[0] : top_left[0] + eye_width] = final_eye
    
            
                #print(eye_width, eye_height)
                #cv2.circle(frame, top_eye, 3, (255, 0, 0), -1)
                #print(face)
                #cv2.imshow("Eye Area", eye_area)
                #cv2.imshow("Eye", eye_sunglass)
                #cv2.imshow("Eye mask", eye_mask)
                #cv2.imshow("Eye mask no mask", eye_area_no_eye)
                #cv2.imshow("Final eye", final_eye)
            
            cv2.imshow("Frame", frame)
    
            key = cv2.waitKey(1)
            if key == 27:
                break
    
    except:
        #print(".")
        action()
if __name__ == '__main__':
    action()
