import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils


def get_index_point(landmark_list):
    index_pos = landmark_list.landmark[8]
    return index_pos.x, index_pos.y


def get_bounding_box(landmark_list):
    x_cords = np.zeros(21)
    y_cords = np.zeros(21)
    for idx, landmark in enumerate(landmark_list.landmark):
        x_cords[idx] = landmark.x
        y_cords[idx] = landmark.y
        # print("landmark ", idx)
        # print(landmark.x, landmark.y)
    x_min = np.amin(x_cords)
    x_max = np.amax(x_cords)
    y_min = np.amin(y_cords)
    y_max = np.amax(y_cords)
    return x_min, y_min, x_max, y_max


def extract_four_fingers_landmark(landmark_list):
    return landmark_list.landmark[8].x, landmark_list.landmark[12].x, landmark_list.landmark[16].x, \
           landmark_list.landmark[20].x


def analyse_fingers(landmark_list):
    # thumb_is_open = False
    first_finger_is_open = False
    second_finger_is_open = False
    third_finger_is_open = False
    fourth_finger_is_open = False

    # right_hand = landmark_list.landmark[5].x < landmark_list.landmark[17].x
    #
    # pseudo_fix_key_point = landmark_list.landmark[2].x
    # if right_hand:
    #     if landmark_list.landmark[3].x < pseudo_fix_key_point and landmark_list.landmark[4].x < pseudo_fix_key_point:
    #         thumb_is_open = True
    # else:
    #     if landmark_list.landmark[3].x > pseudo_fix_key_point and landmark_list.landmark[4].x > pseudo_fix_key_point:
    #         thumb_is_open = True
    pseudo_fix_key_point = landmark_list.landmark[6].y
    if landmark_list.landmark[7].y < pseudo_fix_key_point and landmark_list.landmark[8].y < pseudo_fix_key_point:
        first_finger_is_open = True

    pseudo_fix_key_point = landmark_list.landmark[10].y
    if landmark_list.landmark[11].y < pseudo_fix_key_point and landmark_list.landmark[12].y < pseudo_fix_key_point:
        second_finger_is_open = True

    pseudo_fix_key_point = landmark_list.landmark[14].y
    if landmark_list.landmark[15].y < pseudo_fix_key_point and landmark_list.landmark[16].y < pseudo_fix_key_point:
        third_finger_is_open = True

    pseudo_fix_key_point = landmark_list.landmark[18].y
    if landmark_list.landmark[19].y < pseudo_fix_key_point and landmark_list.landmark[20].y < pseudo_fix_key_point:
        fourth_finger_is_open = True

    if not first_finger_is_open and not second_finger_is_open and not third_finger_is_open \
            and not fourth_finger_is_open:
        return 'fist'
    elif not third_finger_is_open and not fourth_finger_is_open and (
            first_finger_is_open or second_finger_is_open):
        return 'point'
    elif first_finger_is_open and second_finger_is_open and third_finger_is_open \
            and fourth_finger_is_open:
        return 'palm'
    return 'none'
