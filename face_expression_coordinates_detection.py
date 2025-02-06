import cv2
import glob
import dlib
import numpy as np
import pandas as pd

global count
count = 0
global x_points, y_points, distances_list
x_points, y_points, distances_list = [], [], []

landmark_predictor_path = 'shape_predictor_68_face_landmarks.dat_2'  # Ensure this file exists
face_detector_model = dlib.get_frontal_face_detector()
landmark_predictor_model = dlib.shape_predictor(landmark_predictor_path)

def convert_to_array(landmark_instance, dtype="int"):
    coords_array = np.zeros((68, 2), dtype=dtype)
    for idx in range(0, 68):
        coords_array[idx] = (landmark_instance.part(idx).x, landmark_instance.part(idx).y)
    return coords_array

def detect_landmarks(image_input):
    global count
    gray_image = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
    img_width, img_height = gray_image.shape
    target_width = 230
    target_height = int(img_height * (target_width / img_width))
    scale_factor = (target_width / img_width)
    resized_img = cv2.resize(gray_image, (target_height, target_width), interpolation=cv2.INTER_AREA)

    detected_faces = face_detector_model(resized_img)
    if len(detected_faces) > 0:
        for index, face in enumerate(detected_faces):
            landmarks = landmark_predictor_model(resized_img, face)
            
            x_list, y_list, distance_data = [], [], []
            for idx in range(0, 68):
                x_list.append(float(landmarks.part(idx).x))
                y_list.append(float(landmarks.part(idx).y))
            
            mean_x_val = np.mean(x_list)
            mean_y_val = np.mean(y_list)

            x_points.append(x_list)
            y_points.append(y_list)

            # Draw the center point
            cv2.circle(image_input, (int(mean_x_val / scale_factor), int(mean_y_val / scale_factor)), 3, (0, 255, 0), -1)

            center_coordinates = np.asarray((mean_x_val, mean_y_val))
            for x, y in zip(x_list, y_list):
                coord = np.asarray((x, y))
                dist = np.linalg.norm(coord - center_coordinates)
                distance_data.append(float(dist))

            distances_list.append(distance_data)

            # Draw landmark points and connecting lines
            for x, y in zip(x_list, y_list):
                cv2.line(image_input, (int(x / scale_factor), int(y / scale_factor)), (int(mean_x_val / scale_factor), int(mean_y_val / scale_factor)), (0, 0, 255), 1)

            landmark_coordinates = convert_to_array(landmarks)
            for (x, y) in landmark_coordinates:
                cv2.circle(image_input, (int(x / scale_factor), int(y / scale_factor)), 2, (255, 255, 255), -1)

            # Print coordinates
            print("X coordinates of landmarks:", x_list)
            print("Y coordinates of landmarks:", y_list)

    if count == 4:
        landmark_dataframe = pd.DataFrame(columns=['emotion', 'x', 'y', 'distance'])
        for i in range(len(x_points)):
            landmark_dataframe = landmark_dataframe.append({'emotion': 'happy', 'x': x_points[i]}, ignore_index=True)
        landmark_dataframe.to_csv('Facial_Landmarks_Output.csv', encoding='utf-8', index=True)

    count += 1

    # Show the processed image with facial landmarks
    cv2.imshow("Processed Image with Facial Landmarks", image_input)
    cv2.waitKey(0)  # Wait for a key press before closing the window
    cv2.destroyAllWindows()

def execute_main():
    image_paths = glob.glob("*.png")  # Ensure images are available in the directory
    for image_path in image_paths:
        img = cv2.imread(image_path)
        detect_landmarks(img)

if __name__ == "__main__":
    execute_main()
