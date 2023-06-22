import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2


def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes):
  # Extract the face blendshapes category names and scores.
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  fig, ax = plt.subplots(figsize=(12, 12))
  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()

  # Label each bar with values
  for score, patch in zip(face_blendshapes_scores, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

  ax.set_xlabel('Score')
  ax.set_title("Face Blendshapes")
  plt.tight_layout()
  plt.show()


base_options = python.BaseOptions(model_asset_path='/home/matteo/Downloads/face_landmarker.task')
base_options = python.BaseOptions(model_asset_path='/home/matteo/Downloads/archive/train/angry/face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=True,
                                        output_facial_transformation_matrixes=True,
                                        min_face_detection_confidence=0.0,
                                        min_face_presence_confidence=0.0,
                                        min_tracking_confidence=0.0,
                                        num_faces=1)

image_path ="/home/matteo/Downloads/archive/train/fear/Training_318555.jpg"
image_path ="/home/matteo/Downloads/archive/train/angry/Training_143373.jpg"
#image_path = "/home/matteo/newtest.jpg"

img_init = cv2.imread(image_path)
img = cv2.resize(img_init, dsize=(500, 500), interpolation=cv2.INTER_NEAREST)
image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

with vision.FaceLandmarker.create_from_options(options) as landmarker:
    face_landmarker_result = landmarker.detect(image)

#annotated_image = draw_landmarks_on_image(image.numpy_view(), face_landmarker_result)
#plt.imshow(image.numpy_view()); plt.imshow(annotated_image[:, :, 0])
#plt.show()

landmarks = face_landmarker_result.face_landmarks[0] # only one frame
# Canonical (reference) model does not have iris landmarks (last ten),
landmarks_tonp = [[mark.x, mark.y, mark.z] for mark in landmarks]
landmarks_tonp = np.array(landmarks_tonp)

annotated_image = draw_landmarks_on_image(image.numpy_view(), face_landmarker_result)
cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))


plt.figure()
plt.imshow(img)
plt.figure()
plt.scatter(landmarks_tonp[:, 0], landmarks_tonp[:, 1])
plt.show()