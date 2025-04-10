import cv2
import mediapipe as mp
from ultralytics import YOLO

from GazeProcessor import GazeProcessor
from VisualizationOptions import VisualizationOptions
import asyncio

async def gaze_vectors_collected(left, right):
    print(f"left: {left}, right: {right}")

async def main():
    vo = VisualizationOptions()
    gp = GazeProcessor(camera_idx = 1, visualization_options=vo, callback=gaze_vectors_collected)
    await gp.start()

if __name__ == "__main__":
    asyncio.run(main())



# video = cv2.VideoCapture(1)

# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# model = YOLO('yolo11l.pt')

# while True:
#     ret, frame = video.read()
#     if not ret:
#         break

#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(rgb)

#     detections = model.predict(source=frame, classes=[67], stream=True)

#     for r in detections:
#         for box in r.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             conf = box.conf[0].item()
#             if conf > 0.5:
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, f'Phone: {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             left_eye = face_landmarks.landmark[33]
#             right_eye = face_landmarks.landmark[263]
#             nose = face_landmarks.landmark[1]

#             image_height, image_width, _ = frame.shape
#             lx, ly = int(left_eye.x * image_width), int(left_eye.y * image_height)
#             rx, ry = int(right_eye.x * image_width), int(right_eye.y * image_height)
#             nx, ny = int(nose.x * image_width), int(nose.y * image_height)

#             eye_center_x = (lx + rx) // 2
#             eye_center_y = (ly + ry) // 2

#             gaze_vector_x = nx - eye_center_x
#             gaze_vector_y = ny - eye_center_y

#             scale = 2
#             end_point_x = int(eye_center_x + scale * gaze_vector_x)
#             end_point_y = int(eye_center_y + scale * gaze_vector_y)

#             cv2.arrowedLine(frame, (eye_center_x, eye_center_y), (end_point_x, end_point_y), (255, 0, 0), 2)


#     cv2.imshow('Frame', frame)
#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# video.release()
# cv2.destroyAllWindows()
