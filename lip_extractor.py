import cv2
import numpy as np
import mediapipe as mp
import os

LIPS_LANDMARKS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317,
    14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311,
    312, 13, 82, 81, 42, 183, 78
]

def detect_lips(frame, face_mesh):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if not results.multi_face_landmarks:
        return None
    for face_landmarks in results.multi_face_landmarks:
        h, w, _ = frame.shape
        lip_points = np.array([
            (int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h))
            for idx in LIPS_LANDMARKS
        ])
        x, y, w, h = cv2.boundingRect(lip_points)
        if w > 0 and h > 0:
            lips = frame[y:y + h, x:x + w]
            return lips
    return None

def process_video(video_path, output_folder, grid_size=(7, 7), img_size=(256, 256)):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    lip_images = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        lip_frame = detect_lips(frame, face_mesh)
        if lip_frame is not None:
            resized = cv2.resize(lip_frame, img_size)
            lip_images.append(resized)
        if len(lip_images) >= grid_size[0] * grid_size[1]:
            break
    cap.release()

    if not lip_images:
        return None

    while len(lip_images) < grid_size[0] * grid_size[1]:
        lip_images.append(lip_images[0])  # Duplicate images to fill the grid if necessary
    rows = [np.hstack(lip_images[i * grid_size[1]:(i + 1) * grid_size[1]]) for i in range(grid_size[0])]
    grid_image = np.vstack(rows)
    output_path = os.path.join(output_folder, os.path.basename(video_path) + "_lips.jpg")
    cv2.imwrite(output_path, grid_image)
    return output_path