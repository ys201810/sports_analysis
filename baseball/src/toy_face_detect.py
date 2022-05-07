# coding=utf-8
import cv2
import os
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

base_dir = os.path.join(os.getcwd(), '..')


def main():
    # For static images:
    input_dir = os.path.join(base_dir, 'data', 'input')
    output_dir = os.path.join(base_dir, 'data', 'output')

    IMAGE_FILES = [os.path.join(input_dir, 'test1.jpeg')]
    with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5) as face_detection:
        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread(file)
            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
            results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Draw face detections of each face.
            if not results.detections:
                continue
            annotated_image = image.copy()
            for detection in results.detections:
                print('Nose tip:')
                print(mp_face_detection.get_key_point(
                    detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
                mp_drawing.draw_detection(annotated_image, detection)
            cv2.imwrite(os.path.join(output_dir, f'annotated_image_{str(idx)}.png'), annotated_image)


if __name__ == '__main__':
    main()
