import cv2
import mediapipe as mp
import csv
import pandas as pd
import os
import sys

"""
Generate the csv file containing the body keypoints calculated by Mediapipe for each video frame
"""

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

typename = str(sys.argv[1])

FACE_POINTS = 468
HAND_POINTS = 21
write_to_csv = True
undersampled = False
with_face = True
head_landmarks = False
final_dataset_file = ''
points_csv_path = ''

if undersampled and typename == 'train':
    final_dataset_file = '../' + typename + '_dataset_undersampled.csv'
else:
    final_dataset_file = '../' + typename + '_dataset.csv'

root_video_path = r'../GSL_isol'

if undersampled and typename == 'train':
    if with_face:
        points_csv_path = "../points_data/" + typename + "_pose_face_undersampled"
    else:
        points_csv_path = "../points_data/" + typename + "_undersampled"
elif undersampled and typename == 'val':
    print('Cannot write undersampled and validation!')
    sys.exit(1)
elif with_face:
    points_csv_path = "../points_data/" + typename + "_pose_face"
else:
    points_csv_path = "../points_data/" + typename

def extract_points(file_list, filename):
    checksum = 0
    csvlist = []
    frame = 0

    points_with_face = 0
    points_range = None
    if with_face:
        points_range = range(23)
        points_with_face = 23*4
    else:
        points_range = range(11,23)
        points_with_face = (23-11)*4
    
    add_to_checksum_with_head = 0
    if head_landmarks:
        add_to_checksum_with_head = FACE_POINTS*3

    # For static images:
    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=2) as holistic:
        for _, file in enumerate(file_list):
            image = cv2.imread(file)
            results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks != None:
                for point in points_range:
                    
                    csvlist.append(str(results.pose_landmarks.landmark[point].x))
                    csvlist.append(str(results.pose_landmarks.landmark[point].y))
                    csvlist.append(str(results.pose_landmarks.landmark[point].z))
                    csvlist.append(str(results.pose_landmarks.landmark[point].visibility))
            else: csvlist.extend(['0' for _ in range(points_with_face)])

            if(results.left_hand_landmarks != None):
                for point in range(HAND_POINTS):

                    csvlist.append(str(results.left_hand_landmarks.landmark[point].x))
                    csvlist.append(str(results.left_hand_landmarks.landmark[point].y))
                    csvlist.append(str(results.left_hand_landmarks.landmark[point].z))
            else: csvlist.extend(['0' for _ in range(HAND_POINTS*3)])

            if(results.right_hand_landmarks != None):
                for point in range(HAND_POINTS):
                    csvlist.append(str(results.right_hand_landmarks.landmark[point].x))
                    csvlist.append(str(results.right_hand_landmarks.landmark[point].y))
                    csvlist.append(str(results.right_hand_landmarks.landmark[point].z))
            else: csvlist.extend(['0' for _ in range(HAND_POINTS*3)])

            if head_landmarks:
                if results.face_landmarks != None:
                    for point in range(FACE_POINTS):
                        csvlist.append(str(results.face_landmarks.landmark[point].x))
                        csvlist.append(str(results.face_landmarks.landmark[point].y))
                        csvlist.append(str(results.face_landmarks.landmark[point].z))
                else: csvlist.extend(['0' for _ in range(FACE_POINTS*3)])

            # write to csv
            if write_to_csv:
                with open(filename, "a") as csvf:
                    csvwriter = csv.writer(csvf, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csvwriter.writerow(csvlist)

            checksum += (points_with_face+2*(HAND_POINTS*3)+add_to_checksum_with_head) - len(csvlist)
            csvlist.clear()
            frame += 1

    print("Processed csv: {}".format(filename))
    if checksum != 0: print('Points not aligned!')
    else: print("Points per frame: {:d}".format(points_with_face+2*(HAND_POINTS*3)+add_to_checksum_with_head))
    print("Total processed frames: {:d}\n".format(frame))

# prepare video paths
with open(final_dataset_file, "r") as datf:
    csvreader = csv.reader(datf, delimiter=",")
    for row in csvreader:
        video_path = row[0]
        total_path = os.path.join(root_video_path, video_path)
        file_list = [os.path.join(total_path, name) for name in os.listdir(total_path) if os.path.isfile(os.path.join(total_path, name))]
        file_list.sort()
        filename = os.path.join(points_csv_path, video_path.split("/")[0] + "-" + video_path.split("/")[1] + "-" + row[1] + ".csv")

        # create csv
        if write_to_csv:
            open(filename, "w").close()
        
        extract_points(file_list, filename)