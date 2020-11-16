import cv2
import mediapipe as mp

from absl import app, flags
from absl.flags import FLAGS

from kalman.tracker import Tracker
from kalman.detection import Detection
from process_result import get_bounding_box, analyse_fingers, extract_four_fingers_landmark

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

flags.DEFINE_float('chi_sq', 0.95, 'chi-square value for object tracking')


def main(_argv):
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7)

    tracker = Tracker(FLAGS.chi_sq)

    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame

        ret, image = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        tracker.predict()
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                box = get_bounding_box(hand_landmarks)
                finger_landmark = extract_four_fingers_landmark(hand_landmarks)
                class_name = analyse_fingers(hand_landmarks)
                if class_name == 'palm':
                    tracker.update(Detection(box, finger_landmark))
                else:
                    tracker.update(Detection(box))
                font = cv2.FONT_HERSHEY_SIMPLEX
                org = (50, 50)
                font_scale = 1
                color = (255, 0, 0)
                thickness = 2
                image = cv2.putText(image, class_name, org, font,
                                    font_scale, color, thickness, cv2.LINE_AA)
        else:
            tracker.update(None)

        if tracker.track is not None and tracker.track.is_confirmed():
            pointer = tracker.track.mean
            pointer_x = int(pointer[0] * image.shape[1])
            pointer_y = int(pointer[1] * image.shape[0])
            cv2.circle(image, (pointer_x, pointer_y), 30, (0, 0, 255), -1)

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    hands.close()
    cap.release()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
