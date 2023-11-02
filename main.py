import cv2 as cv
import sys
import time
import math
import numpy as np
import random as rng
from scipy import stats 
import kociemba
from rubik_solver import utils

video = cv.VideoCapture(0)
def rotate_cw(face):
    final = np.copy(face)
    final[0] = face[6]
    final[1] = face[3]
    final[2] = face[0]
    final[3] = face[7]
    final[4] = face[4]
    final[5] = face[1]
    final[6] = face[8]
    final[7] = face[5]
    final[8] = face[2]
    return final

def rotate_ccw(face):
    final = np.copy(face)
    final[8] = face[6]
    final[7] = face[3]
    final[6] = face[0]
    final[5] = face[7]
    final[4] = face[4]
    final[3] = face[1]
    final[2] = face[8]
    final[1] = face[5]
    final[0] = face[2]
    return final

def right_cw(video,up_face,right_face,front_face,down_face,left_face,back_face):
    print("Next Move: R Clockwise")
    temp = np.copy(front_face)
    front_face[2] = down_face[2]
    front_face[5] = down_face[5]
    front_face[8] = down_face[8]
    down_face[2] = back_face[6]
    down_face[5] = back_face[3]
    down_face[8] = back_face[0]
    back_face[0] = up_face[0]
    back_face[3] = up_face[5]
    back_face[6] = up_face[2]
    up_face[2] = temp[2]
    up_face[5] = temp[5]
    up_face[8] = temp[8]
    right_face = rotate_cw(right_face)

    print(front_face)
    faces = []
    while True:
        is_ok, frame = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, blob_colors = detect_face(frame)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                detected_face = stats.mode(face_array, keepdims=False)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:  #hai mang co phan tu va kt giong nhau
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face
                elif np.array_equal(detected_face,temp) == True:
                    centroid1 = blob_colors[8]  #
                    centroid2 = blob_colors[2]
                    point1 = (centroid1[5]+int((centroid1[7]/2)), centroid1[6]+int((centroid1[7]/2)))
                    point2 = (centroid2[5]+int((centroid2[8]/2)), centroid2[6]+int((centroid2[8]/2)))
                    cv.arrowedLine(frame, point1, point2, (0, 0, 0), 7, tipLength = 0.2)
                    cv.arrowedLine(frame, point1, point2, (0, 0, 255), 4, tipLength=0.2)
        cv.imshow("Output Image", frame)
        key_pressed = cv.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break
def right_ccw(video,up_face,right_face,front_face,down_face,left_face,back_face):
    print("Next Move: R CounterClockwise")
    temp = np.copy(front_face)
    front_face[2] = up_face[2]
    front_face[5] = up_face[5]
    front_face[8] = up_face[8]
    up_face[2] = back_face[6]
    up_face[5] = back_face[3]
    up_face[8] = back_face[0]
    back_face[0] = down_face[8]
    back_face[3] = down_face[5]
    back_face[6] = down_face[2]
    down_face[2] = temp[2]
    down_face[5] = temp[5]
    down_face[8] = temp[8]
    right_face = rotate_ccw(right_face)
    # front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, frame = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, blob_colors = detect_face(frame)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                detected_face = stats.mode(face_array, keepdims=False)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face
                elif np.array_equal(detected_face,temp) == True:
                    centroid1 = blob_colors[2]
                    centroid2 = blob_colors[8]
                    point1 = (centroid1[5]+int((centroid1[7]/2)), centroid1[6]+int((centroid1[7]/2)))
                    point2 = (centroid2[5]+int((centroid2[8]/2)), centroid2[6]+int((centroid2[8]/2)))
                    cv.arrowedLine(frame, point1, point2, (0, 0, 0), 7, tipLength = 0.2)
                    cv.arrowedLine(frame, point1, point2, (0, 0, 255), 4, tipLength=0.2)
        cv.imshow("Output Image", frame)
        key_pressed = cv.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

def left_cw(video,up_face,right_face,front_face,down_face,left_face,back_face):
    print("Next Move: L Clockwise")
    temp = np.copy(front_face)
    front_face[0] = up_face[0]
    front_face[3] = up_face[3]
    front_face[6] = up_face[6]
    up_face[0] = back_face[8]
    up_face[3] = back_face[5]
    up_face[6] = back_face[2]
    back_face[2] = down_face[6]
    back_face[5] = down_face[3]
    back_face[8] = down_face[0]
    down_face[0] = temp[0]
    down_face[3] = temp[3]
    down_face[6] = temp[6]
    left_face = rotate_cw(left_face)
    #front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, frame = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, blob_colors = detect_face(frame)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array, keepdims=False)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face
                elif np.array_equal(detected_face,temp) == True:
                    centroid1 = blob_colors[0]
                    centroid2 = blob_colors[6]
                    point1 = (centroid1[5]+int((centroid1[7]/2)), centroid1[6]+int((centroid1[7]/2)))
                    point2 = (centroid2[5]+int((centroid2[8]/2)), centroid2[6]+int((centroid2[8]/2)))
                    cv.arrowedLine(frame, point1, point2, (0, 0, 0), 7, tipLength = 0.2)
                    cv.arrowedLine(frame, point1, point2, (0, 0, 255), 4, tipLength=0.2)
        cv.imshow("Output Image", frame)
        key_pressed = cv.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

def left_ccw(video,up_face,right_face,front_face,down_face,left_face,back_face):
    print("Next Move: L CounterClockwise")
    temp = np.copy(front_face)
    front_face[0] = down_face[0]
    front_face[3] = down_face[3]
    front_face[6] = down_face[6]
    down_face[0] = back_face[8]
    down_face[3] = back_face[5]
    down_face[6] = back_face[2]
    back_face[2] = up_face[6]
    back_face[5] = up_face[3]
    back_face[8] = up_face[0]
    up_face[0] = temp[0]
    up_face[3] = temp[3]
    up_face[6] = temp[6]
    left_face = rotate_ccw(left_face)
    #front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, frame = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, blob_colors = detect_face(frame)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array, keepdims=False)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face
                elif np.array_equal(detected_face,temp) == True:
                    centroid1 = blob_colors[6]
                    centroid2 = blob_colors[0]
                    point1 = (centroid1[5]+int((centroid1[7]/2)), centroid1[6]+int((centroid1[7]/2)))
                    point2 = (centroid2[5]+int((centroid2[8]/2)), centroid2[6]+int((centroid2[8]/2)))
                    cv.arrowedLine(frame, point1, point2, (0, 0, 0), 7, tipLength = 0.2)
                    cv.arrowedLine(frame, point1, point2, (0, 0, 255), 4, tipLength=0.2)
        cv.imshow("Output Image", frame)
        key_pressed = cv.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

def front_cw(video,up_face,right_face,front_face,down_face,left_face,back_face):
    print(front_face)
    print("Next Move: F Clockwise")
    temp1 = np.copy(front_face)
    temp = np.copy(up_face)
    front_face = rotate_cw(front_face)
    temp2 = np.copy(front_face)
    if np.array_equal(temp2, temp1) == True:
        [up_face, right_face, front_face, down_face, left_face, back_face] = turn_to_right(video, up_face, right_face, front_face, down_face, left_face, back_face)
        [up_face, right_face, front_face, down_face, left_face, back_face] = left_cw(video, up_face, right_face, front_face, down_face, left_face, back_face)
        [up_face, right_face, front_face, down_face, left_face, back_face] = turn_to_front(video,up_face, right_face, front_face, down_face, left_face, back_face)
        return up_face, right_face, front_face, down_face, left_face, back_face
    up_face[8] = left_face[2]
    up_face[7] = left_face[5]
    up_face[6] = left_face[8]
    left_face[2] = down_face[0]
    left_face[5] = down_face[1]
    left_face[8] = down_face[2]
    down_face[2] = right_face[0]
    down_face[1] = right_face[3]
    down_face[0] = right_face[6]
    right_face[0] = temp[6]
    right_face[3] = temp[7]
    right_face[6] = temp[8]

    print(front_face)
    faces = []
    while True:

        is_ok, frame = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, blob_colors = detect_face(frame)
        print('face',face)
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                detected_face = stats.mode(face_array, keepdims=False)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face
                elif np.array_equal(detected_face,temp1) == True:
                    centroid1 = blob_colors[8]
                    centroid2 = blob_colors[6]
                    centroid3 = blob_colors[0]
                    centroid4 = blob_colors[2]
                    point1 = (centroid1[5] + int((centroid1[7] / 4)), centroid1[6] + int((centroid1[7] / 2)))
                    point2 = (centroid2[5] + int((3 * centroid2[8] / 4)), centroid2[6] + int((centroid2[8] / 2)))
                    point3 = (centroid2[5] + int((centroid2[7] / 2)), centroid2[6] + int((centroid2[7] / 4)))
                    point4 = (centroid3[5] + int((centroid3[8] / 2)), centroid3[6] + int((3 * centroid3[8] / 4)))
                    point5 = (centroid3[5] + int((3 * centroid3[8] / 4)), centroid3[6] + int((centroid3[8] / 2)))
                    point6 = (centroid4[5] + int((centroid4[8] / 4)), centroid4[6] + int((centroid4[8] / 2)))
                    point7 = (centroid4[5] + int((centroid4[8] / 2)), centroid4[6] + int((3 * centroid4[8] / 4)))
                    point8 = (centroid1[5] + int((centroid1[8] / 2)), centroid1[6] + int((centroid1[8] / 4)))
                    cv.arrowedLine(frame, point1, point2, (0, 0, 0), 7, tipLength = 0.2)
                    cv.arrowedLine(frame, point3, point4, (0, 0, 0), 7, tipLength = 0.2)
                    cv.arrowedLine(frame, point5, point6, (0, 0, 0), 7, tipLength = 0.2)
                    cv.arrowedLine(frame, point7, point8, (0, 0, 0), 7, tipLength = 0.2)
                    cv.arrowedLine(frame, point1, point2, (0, 0, 255), 4, tipLength=0.2)
                    cv.arrowedLine(frame, point3, point4, (0, 0, 255), 4, tipLength=0.2)
                    cv.arrowedLine(frame, point5, point6, (0, 0, 255), 4, tipLength=0.2)
                    cv.arrowedLine(frame, point7, point8, (0, 0, 255), 4, tipLength=0.2)
        cv.imshow("Output Image", frame)
        key_pressed = cv.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

def front_ccw(video,up_face,right_face,front_face,down_face,left_face,back_face):
    print("Next Move: F CounterClockwise")
    temp = np.copy(up_face)
    temp1 = np.copy(front_face)
    front_face = rotate_ccw(front_face)
    temp2 = np.copy(front_face)
    if np.array_equal(temp2,temp1) == True:
            [up_face, right_face, front_face, down_face, left_face, back_face] = turn_to_right(video,up_face,right_face,front_face,down_face,left_face,back_face)
            [up_face, right_face, front_face, down_face, left_face, back_face] = left_ccw(video,up_face,right_face,front_face,down_face,left_face,back_face)
            [up_face, right_face, front_face, down_face, left_face, back_face] = turn_to_front(video,up_face,right_face,front_face,down_face,left_face,back_face)
            return up_face,right_face,front_face,down_face,left_face,back_face
    up_face[6] = right_face[0]
    up_face[7] = right_face[3]
    up_face[8] = right_face[6]
    right_face[0] = down_face[2]
    right_face[3] = down_face[1]
    right_face[6] = down_face[0]
    down_face[0] = left_face[2]
    down_face[1] = left_face[5]
    down_face[2] = left_face[8]
    left_face[8] = temp[6]
    left_face[5] = temp[7]
    left_face[2] = temp[8]

    #front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, frame = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, blob_colors = detect_face(frame)
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                detected_face = stats.mode(face_array, keepdims=False)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face
                elif np.array_equal(detected_face,temp1) == True:
                    centroid1 = blob_colors[2]
                    centroid2 = blob_colors[0]
                    centroid3 = blob_colors[6]
                    centroid4 = blob_colors[8]
                    point1 = (centroid1[5] + int((centroid1[7] / 4)), centroid1[6] + int((centroid1[7] / 2)))
                    point2 = (centroid2[5] + int((3 * centroid2[8]/4)), centroid2[6] + int((centroid2[8] / 2)))
                    point3 = (centroid2[5] + int((centroid2[7] / 2)), centroid2[6] + int((3 * centroid2[7] / 4)))
                    point4 = (centroid3[5] + int((centroid3[8] / 2)), centroid3[6] + int((centroid3[8] / 4)))
                    point5 = (centroid3[5] + int((3 * centroid3[8] / 4)), centroid3[6] + int((centroid3[8] / 2)))
                    point6 = (centroid4[5] + int((centroid4[8] / 4)), centroid4[6] + int((centroid4[8] / 2)))
                    point7 = (centroid4[5] + int((centroid4[8] / 2)), centroid4[6] + int((centroid4[8] / 4)))
                    point8 = (centroid1[5] + int((centroid1[8] / 2)), centroid1[6] + int((3 * centroid1[8] / 4)))
                    cv.arrowedLine(frame, point1, point2, (0, 0, 0), 7, tipLength = 0.2)
                    cv.arrowedLine(frame, point3, point4, (0, 0, 0), 7, tipLength = 0.2)
                    cv.arrowedLine(frame, point5, point6, (0, 0, 0), 7, tipLength = 0.2)
                    cv.arrowedLine(frame, point7, point8, (0, 0, 0), 7, tipLength = 0.2)
                    cv.arrowedLine(frame, point1, point2, (0, 0, 255), 4, tipLength=0.2)
                    cv.arrowedLine(frame, point3, point4, (0, 0, 255), 4, tipLength=0.2)
                    cv.arrowedLine(frame, point5, point6, (0, 0, 255), 4, tipLength=0.2)
                    cv.arrowedLine(frame, point7, point8, (0, 0, 255), 4, tipLength=0.2)
        cv.imshow("Output Image", frame)
        key_pressed = cv.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

def back_cw(video,up_face,right_face,front_face,down_face,left_face,back_face):
    print("Next Move: B Clockwise")
    temp = np.copy(up_face)
    up_face[0] = right_face[2]
    up_face[1] = right_face[5]
    up_face[2] = right_face[8]
    right_face[8] = down_face[6]
    right_face[5] = down_face[7]
    right_face[2] = down_face[8]
    down_face[6] = left_face[0]
    down_face[7] = left_face[3]
    down_face[8] = left_face[0, 6]
    left_face[0] = temp[2]
    left_face[3] = temp[1]
    left_face[6] = temp[0]
    back_face = rotate_cw(back_face)
    #front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, frame = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, blob_colors = detect_face(frame)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array, keepdims=False)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face
        cv.imshow("Output Image", frame)
        key_pressed = cv.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

def back_ccw(video,up_face,right_face,front_face,down_face,left_face,back_face):
    print("Next Move: B CounterClockwise")
    temp = np.copy(up_face)
    up_face[2] = left_face[0]
    up_face[1] = left_face[3]
    up_face[0] = left_face[6]
    left_face[0] = down_face[6]
    left_face[3] = down_face[7]
    left_face[6] = down_face[8]
    down_face[6] = right_face[0, 8]
    down_face[7] = right_face[0, 5]
    down_face[8] = right_face[0, 2]
    right_face[2] = temp[0]
    right_face[5] = temp[1]
    right_face[8] = temp[2]
    back_face = rotate_ccw(back_face)
    #front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, frame = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, blob_colors = detect_face(frame)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array, keepdims=False)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face
        cv.imshow("Output Image", frame)
        key_pressed = cv.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

def up_cw(video,up_face,right_face,front_face,down_face,left_face,back_face):
    print("Next Move: U Clockwise")
    temp = np.copy(front_face)
    front_face[0] = right_face[0]
    front_face[1] = right_face[1]
    front_face[2] = right_face[2]
    right_face[0] = back_face[0]
    right_face[1] = back_face[1]
    right_face[2] = back_face[2]
    back_face[0] = left_face[0]
    back_face[1] = left_face[1]
    back_face[2] = left_face[2]
    left_face[0] = temp[0]
    left_face[1] = temp[1]
    left_face[2] = temp[2]
    up_face = rotate_cw(up_face)
    #front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, frame = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, blob_colors = detect_face(frame)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array, keepdims=False)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face
                elif np.array_equal(detected_face,temp) == True:
                    centroid1 = blob_colors[2]
                    centroid2 = blob_colors[0]
                    point1 = (centroid1[5]+int((centroid1[7]/2)), centroid1[6]+int((centroid1[7]/2)))
                    point2 = (centroid2[5]+int((centroid2[8]/2)), centroid2[6]+int((centroid2[8]/2)))
                    cv.arrowedLine(frame, point1, point2, (0, 0, 0), 7, tipLength = 0.2)
                    cv.arrowedLine(frame, point1, point2, (0, 0, 255), 4, tipLength=0.2)
        cv.imshow("Output Image", frame)
        key_pressed = cv.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

def up_ccw(video,up_face,right_face,front_face,down_face,left_face,back_face):
    print("Next Move: U CounterClockwise")
    temp = np.copy(front_face)
    front_face[0] = left_face[0]
    front_face[1] = left_face[1]
    front_face[2] = left_face[2]
    left_face[0] = back_face[0]
    left_face[1] = back_face[1]
    left_face[2] = back_face[2]
    back_face[0] = right_face[0]
    back_face[1] = right_face[1]
    back_face[2] = right_face[2]
    right_face[0] = temp[0]
    right_face[1] = temp[1]
    right_face[2] = temp[2]
    up_face = rotate_ccw(up_face)
    #front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, frame = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, blob_colors = detect_face(frame)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array, keepdims=False)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face
                elif np.array_equal(detected_face,temp) == True:
                    centroid1 = blob_colors[0]
                    centroid2 = blob_colors[2]
                    point1 = (centroid1[5]+int((centroid1[7]/2)), centroid1[6]+int((centroid1[7]/2)))
                    point2 = (centroid2[5]+int((centroid2[8]/2)), centroid2[6]+int((centroid2[8]/2)))
                    cv.arrowedLine(frame, point1, point2, (0, 0, 0), 7, tipLength = 0.2)
                    cv.arrowedLine(frame, point1, point2, (0, 0, 255), 4, tipLength=0.2)
        cv.imshow("Output Image", frame)
        key_pressed = cv.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

def down_cw(video,up_face,right_face,front_face,down_face,left_face,back_face):
    print("Next Move: D Clockwise")
    temp = np.copy(front_face)
    front_face[6] = left_face[6]
    front_face[7] = left_face[7]
    front_face[8] = left_face[8]
    left_face[6] = back_face[6]
    left_face[7] = back_face[7]
    left_face[8] = back_face[8]
    back_face[6] = right_face[6]
    back_face[7] = right_face[7]
    back_face[8] = right_face[8]
    right_face[6] = temp[6]
    right_face[7] = temp[7]
    right_face[8] = temp[8]
    down_face = rotate_cw(down_face)
    #front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, frame = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, blob_colors = detect_face(frame)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array, keepdims=False)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face
                elif np.array_equal(detected_face,temp) == True:
                    centroid1 = blob_colors[6]
                    centroid2 = blob_colors[8]
                    point1 = (centroid1[5]+int((centroid1[7]/2)), centroid1[6]+int((centroid1[7]/2)))
                    point2 = (centroid2[5]+int((centroid2[8]/2)), centroid2[6]+int((centroid2[8]/2)))
                    cv.arrowedLine(frame, point1, point2, (0, 0, 0), 7, tipLength = 0.2)
                    cv.arrowedLine(frame, point1, point2, (0, 0, 255), 4, tipLength=0.2)
        cv.imshow("Output Image", frame)
        key_pressed = cv.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

def down_ccw(video,up_face,right_face,front_face,down_face,left_face,back_face):
    print("Next Move: D CounterClockwise")
    temp = np.copy(front_face)
    front_face[6] = right_face[6]
    front_face[7] = right_face[7]
    front_face[8] = right_face[8]
    right_face[6] = back_face[6]
    right_face[7] = back_face[7]
    right_face[8] = back_face[8]
    back_face[6] = left_face[6]
    back_face[7] = left_face[7]
    back_face[8] = left_face[8]
    left_face[6] = temp[6]
    left_face[7] = temp[7]
    left_face[8] = temp[8]
    down_face = rotate_ccw(down_face)
    #front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, frame = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, blob_colors = detect_face(frame)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array, keepdims=False)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face
                elif np.array_equal(detected_face,temp) == True:
                    centroid1 = blob_colors[8]
                    centroid2 = blob_colors[6]
                    point1 = (centroid1[5]+int((centroid1[7]/2)), centroid1[6]+int((centroid1[7]/2)))
                    point2 = (centroid2[5]+int((centroid2[8]/2)), centroid2[6]+int((centroid2[8]/2)))
                    cv.arrowedLine(frame, point1, point2, (0, 0, 0), 7, tipLength = 0.2)
                    cv.arrowedLine(frame, point1, point2, (0, 0, 255), 4, tipLength=0.2)
        cv.imshow("Output Image", frame)
        key_pressed = cv.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

def turn_to_right(video,up_face,right_face,front_face,down_face,left_face,back_face):
    print("Next Move: Show Right Face")
    temp = np.copy(front_face)
    front_face = np.copy(right_face)
    right_face = np.copy(back_face)
    back_face = np.copy(left_face)
    left_face = np.copy(temp)
    up_face = rotate_cw(up_face)
    down_face = rotate_ccw(down_face)
    #front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, frame = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, blob_colors = detect_face(frame)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array, keepdims=False)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face
                elif np.array_equal(detected_face,temp) == True:
                    centroid1 = blob_colors[8]
                    centroid2 = blob_colors[6]
                    centroid3 = blob_colors[5]
                    centroid4 = blob_colors[3]
                    centroid5 = blob_colors[2]
                    centroid6 = blob_colors[0]
                    point1 = (centroid1[5] + int((centroid1[7] / 2)), centroid1[6] + int((centroid1[7] / 2)))
                    point2 = (centroid2[5] + int((centroid2[8] / 2)), centroid2[6] + int((centroid2[8] / 2)))
                    point3 = (centroid3[5] + int((centroid3[7] / 2)), centroid3[6] + int((centroid3[7] / 2)))
                    point4 = (centroid4[5] + int((centroid4[8] / 2)), centroid4[6] + int((centroid4[8] / 2)))
                    point5 = (centroid5[5] + int((centroid5[7] / 2)), centroid5[6] + int((centroid5[7] / 2)))
                    point6 = (centroid6[5] + int((centroid6[8] / 2)), centroid6[6] + int((centroid6[8] / 2)))
                    cv.arrowedLine(frame, point1, point2, (0, 0, 0), 7, tipLength = 0.2)
                    cv.arrowedLine(frame, point3, point4, (0, 0, 0), 7, tipLength = 0.2)
                    cv.arrowedLine(frame, point5, point6, (0, 0, 0), 7, tipLength = 0.2)
                    cv.arrowedLine(frame, point1, point2, (0, 0, 255), 4, tipLength=0.2)
                    cv.arrowedLine(frame, point3, point4, (0, 0, 255), 4, tipLength=0.2)
                    cv.arrowedLine(frame, point5, point6, (0, 0, 255), 4, tipLength=0.2)
        cv.imshow("Output Image", frame)
        key_pressed = cv.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

def turn_to_front(video,up_face,right_face,front_face,down_face,left_face,back_face):
    print("Next Move: Show Front Face")
    temp = np.copy(front_face)
    front_face = np.copy(left_face)
    left_face = np.copy(back_face)
    back_face = np.copy(right_face)
    right_face = np.copy(temp)
    up_face = rotate_ccw(up_face)
    down_face = rotate_cw(down_face)
    #front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, frame = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, blob_colors = detect_face(frame)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                detected_face = stats.mode(face_array, keepdims=False)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face
                elif np.array_equal(detected_face,temp) == True:
                    centroid1 = blob_colors[6]
                    centroid2 = blob_colors[8]
                    centroid3 = blob_colors[3]
                    centroid4 = blob_colors[5]
                    centroid5 = blob_colors[0]
                    centroid6 = blob_colors[2]
                    point1 = (centroid1[5] + int((centroid1[7] / 2)), centroid1[6] + int((centroid1[7] / 2)))
                    point2 = (centroid2[5] + int((centroid2[8] / 2)), centroid2[6] + int((centroid2[8] / 2)))
                    point3 = (centroid3[5] + int((centroid3[7] / 2)), centroid3[6] + int((centroid3[7] / 2)))
                    point4 = (centroid4[5] + int((centroid4[8] / 2)), centroid4[6] + int((centroid4[8] / 2)))
                    point5 = (centroid5[5] + int((centroid5[7] / 2)), centroid5[6] + int((centroid5[7] / 2)))
                    point6 = (centroid6[5] + int((centroid6[8] / 2)), centroid6[6] + int((centroid6[8] / 2)))
                    cv.arrowedLine(frame, point1, point2, (0, 0, 0), 7, tipLength = 0.2)
                    cv.arrowedLine(frame, point3, point4, (0, 0, 0), 7, tipLength = 0.2)
                    cv.arrowedLine(frame, point5, point6, (0, 0, 0), 7, tipLength = 0.2)
                    cv.arrowedLine(frame, point1, point2, (0, 0, 255), 4, tipLength=0.2)
                    cv.arrowedLine(frame, point3, point4, (0, 0, 255), 4, tipLength=0.2)
                    cv.arrowedLine(frame, point5, point6, (0, 0, 255), 4, tipLength=0.2)
        cv.imshow("Output Image", frame)
        key_pressed = cv.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

def concat(up_face,left_face,front_face,right_face,back_face,down_face):
    solution = np.concatenate((up_face, left_face), axis=None)
    solution = np.concatenate((solution,front_face), axis=None)
    solution = np.concatenate((solution, right_face), axis=None)
    solution = np.concatenate((solution, back_face), axis=None)
    solution = np.concatenate((solution, down_face), axis=None)
    return solution
def detect_face(frame):

    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(2,2))
    gray = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel)
    gray = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel)
    gray = cv.adaptiveThreshold(gray,25,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,15,0)
    cv.imshow('gray',gray)
    try:
         _, contours, hierarchy = cv.findContours(gray,cv.RETR_CCOMP,cv.CHAIN_APPROX_NONE)
    except:
         contours, hierarchy = cv.findContours(gray,cv.RETR_CCOMP,cv.CHAIN_APPROX_NONE)


    i = 0
    tmp = []
    contour_id = 0
    #print(len(contours))
    count = 0
    blob_colors = []
    for contour in contours:
        A1 = cv.contourArea(contour)
        contour_id = contour_id + 1

        if A1 < 3000 and A1 > 1000:
            perimeter = cv.arcLength(contour, True)
            epsilon = 0.01 * perimeter
            approx = cv.approxPolyDP(contour, epsilon, True)
            hull = cv.convexHull(contour)
            if cv.norm(((perimeter / 4) * (perimeter / 4)) - A1) < 150 or ((3.14*(perimeter / 4) * (perimeter / 4)) - A1) < 150:  #hình vuông và hình tròn
                count = count + 1
                x, y, w, h = cv.boundingRect(contour)
                val = (50*y) + (10*x)
                y0=int(y+0.3*h)
                y1=int(y+0.7*h)
                x0=int(x+0.3*w)
                x1=int(x+0.7*w)
                blob_color = np.array(cv.mean(frame[y0:y1,x0:x1])).astype(int)  
                cv.drawContours(frame,[contour],0,(255, 255, 0),2)
                cv.drawContours(frame, [approx], 0, (255, 255, 0), 2)
                blob_color = np.append(blob_color, val)
                blob_color = np.append(blob_color, x)
                blob_color = np.append(blob_color, y)
                blob_color = np.append(blob_color, w)
                blob_color = np.append(blob_color, h)
                blob_colors.append(blob_color)
    if len(blob_colors) > 0:
        blob_colors = np.asarray(blob_colors)
        blob_colors = blob_colors[blob_colors[:, 4].argsort()]

    face = np.array([0,0,0,0,0,0,0,0,0])
    if len(blob_colors) == 9:
        #print(blob_colors)
        for i in range(9):
            #print(blob_colors[i])
            if blob_colors[i][0] > 130 and blob_colors[i][1] > 130 and blob_colors[i][2] > 130:
                face[i] = 1     #white                
            elif blob_colors[i][0] < 100 and blob_colors[i][1] > 120 and blob_colors[i][2] > 120 and np.abs(blob_colors[i][1]-blob_colors[i][2])<30:
                face[i] = 2   #yellow                 
            elif blob_colors[i][0] > blob_colors[i][1] and blob_colors[i][1] > blob_colors[i][2]:
                face[i] = 3   #blue                
            elif blob_colors[i][1] > blob_colors[i][0] and blob_colors[i][1] > blob_colors[i][2] and np.abs(blob_colors[i][0] - blob_colors[i][2]) < 50:
                face[i] = 4   #green                
            elif blob_colors[i][2] > blob_colors[i][0] and blob_colors[i][2] > blob_colors[i][1] and np.abs(blob_colors[i][0] - blob_colors[i][1]) < 30 and blob_colors[i][0] < 140:
                face[i] = 5   #red
            elif blob_colors[i][1] < blob_colors[i][2] and blob_colors[i][0] < blob_colors[i][1] and blob_colors[i][2] > 120:      
                face[i] = 6  #orange
        if np.count_nonzero(face) == 9:
            return face, blob_colors
        else:
            return [0,0], blob_colors
    else:
        return [0,0,0], blob_colors
        #break

faces = []
cube_char =[]
old_face = 7
luu =[]
left_face=[]
down_face=[]
front_face=[]
right_face=[]
up_face=[]
back_face=[]
tmp_solved=[]
while True:
    is_ok, frame = video.read()

    if not is_ok:
        print("Cannot read video source")
        sys.exit()
        
    face, blob_colors = detect_face(frame)
    if len(face) == 9:
        # print(face)
        if (old_face != face[4]):
            old_face = face[4]
            cube_char =np.append(cube_char,face)
            if (len(cube_char) == 9 and cube_char[4] == 2.0):
                up_face = np.append(up_face,face).astype(int)
            elif (len(cube_char) == 18 and cube_char[13] == 3.0):
                left_face = np.append(left_face,face).astype(int)
            elif (len(cube_char) == 27 and cube_char[22] == 5.0):
                front_face = np.append(front_face,face).astype(int)
            elif (len(cube_char) == 36 and cube_char[31] == 4.0):
                right_face = np.append(right_face,face).astype(int)
            elif (len(cube_char) == 45 and cube_char[40] == 6.0):
                back_face = np.append(back_face,face).astype(int)
            elif (len(cube_char) == 54 and cube_char[49] == 1.0):
                down_face = np.append(down_face,face).astype(int)
            print('up_face : ',up_face)
            print('left_face : ',left_face)
            print('front_face : ',front_face)
            print('right_face : ',right_face)
            print('back_face : ',back_face)
            print('down_face : ',down_face)
            solver = concat(up_face,left_face,front_face,right_face,back_face,down_face)  # 54 số màu hiện tại
            if (len(solver) == 54):
                for index in range(len(solver)):
                    if ( cube_char[index] == 1.0):
                        luu = np.append(luu,'w')
                        luu = ''.join(luu)
                    elif ( cube_char[index] == 2.0):
                        luu = np.append(luu,'y')
                        luu = ''.join(luu)
                    elif ( cube_char[index] == 3.0):
                        luu = np.append(luu,'b')
                        luu = ''.join(luu)
                    elif ( cube_char[index] == 4.0):
                        luu = np.append(luu,'g')
                        luu = ''.join(luu)
                    elif ( cube_char[index] == 5.0):
                        luu = np.append(luu,'r')
                        luu = ''.join(luu)
                    elif ( cube_char[index] == 6.0):
                        luu = np.append(luu,'o')
                        luu = ''.join(luu)
                print(luu)
                
                solved = utils.solve(luu, 'Kociemba')
                tmp_solved = solved
                print(solved)
           
    for step in tmp_solved:
        if step == "R":
            [up_face, right_face, front_face, down_face, left_face, back_face] = right_cw(video,up_face,right_face,front_face,down_face,left_face,back_face)
            
        elif step == "R'":
            [up_face, right_face, front_face, down_face, left_face, back_face] = right_ccw(video ,  up_face, right_face, front_face, down_face, left_face, back_face)
            
        elif step == "R2":
            [up_face, right_face, front_face, down_face, left_face, back_face] = right_cw(video,  up_face, right_face, front_face, down_face, left_face, back_face)
            [up_face, right_face, front_face, down_face, left_face, back_face] = right_cw(video,  up_face, right_face, front_face, down_face, left_face, back_face)
            
        elif step == "L":
            [up_face, right_face, front_face, down_face, left_face, back_face] = left_cw(video,  up_face, right_face, front_face, down_face, left_face, back_face)
            
        elif step == "L'":
            [up_face, right_face, front_face, down_face, left_face, back_face] = left_ccw(video,  up_face, right_face, front_face, down_face, left_face, back_face)
            
        elif step == "L2":
            [up_face, right_face, front_face, down_face, left_face, back_face] = left_cw(video,  up_face, right_face, front_face, down_face, left_face, back_face)
            [up_face, right_face, front_face, down_face, left_face, back_face] = left_cw(video,  up_face, right_face, front_face, down_face, left_face, back_face)
            
        elif step == "F":
            [up_face, right_face, front_face, down_face, left_face, back_face] = front_cw(video,  up_face, right_face, front_face, down_face, left_face, back_face)
            
        elif step == "F'":
            [up_face, right_face, front_face, down_face, left_face, back_face] = front_ccw(video,  up_face, right_face, front_face, down_face, left_face, back_face)
            
        elif step == "F2":
            [up_face, right_face, front_face, down_face, left_face, back_face] = front_cw(video,  up_face, right_face, front_face, down_face, left_face, back_face)
            [up_face, right_face, front_face, down_face, left_face, back_face] = front_cw(video,  up_face, right_face, front_face, down_face, left_face, back_face)
            
        elif step == "B":
            [up_face, right_face, front_face, down_face, left_face, back_face] = turn_to_right(video,  up_face, right_face, front_face, down_face, left_face, back_face)
            [up_face, right_face, front_face, down_face, left_face, back_face] = right_cw(video,  up_face, right_face, front_face, down_face, left_face, back_face)
            [up_face, right_face, front_face, down_face, left_face, back_face] = turn_to_front(video,  up_face, right_face, front_face, down_face, left_face, back_face)
            
        elif step == "B'":
             [up_face, right_face, front_face, down_face, left_face, back_face] = turn_to_right(video,  up_face, right_face, front_face, down_face, left_face, back_face)
            [up_face, right_face, front_face, down_face, left_face, back_face] = right_ccw(video,  up_face, right_face, front_face, down_face, left_face, back_face)
            [up_face, right_face, front_face, down_face, left_face, back_face] = turn_to_front(video,  up_face, right_face, front_face, down_face, left_face, back_face)
        elif step == "B2":
            [up_face, right_face, front_face, down_face, left_face, back_face] = turn_to_right(video,  up_face, right_face, front_face, down_face, left_face, back_face)
            [up_face, right_face, front_face, down_face, left_face, back_face] = right_cw(video,  up_face, right_face, front_face, down_face, left_face, back_face)
            [up_face, right_face, front_face, down_face, left_face, back_face] = right_cw(video,  up_face, right_face, front_face, down_face, left_face, back_face)
            [up_face, right_face, front_face, down_face, left_face, back_face] = turn_to_front(video,  up_face, right_face, front_face, down_face, left_face, back_face)
            
        elif step == "U":
            [up_face, right_face, front_face, down_face, left_face, back_face] = up_cw(video,  up_face, right_face, front_face, down_face, left_face, back_face)
            
        elif step == "U'":
            [up_face, right_face, front_face, down_face, left_face, back_face] = up_ccw(video,  up_face, right_face, front_face, down_face, left_face, back_face)
            
        elif step == "U2":
            [up_face, right_face, front_face, down_face, left_face, back_face] = up_cw(video,  up_face, right_face, front_face, down_face, left_face, back_face)
            [up_face, right_face, front_face, down_face, left_face, back_face] = up_cw(video,  up_face, right_face, front_face, down_face, left_face, back_face)
            
        elif step == "D":
            [up_face, right_face, front_face, down_face, left_face, back_face] = down_cw(video,  up_face, right_face, front_face, down_face, left_face, back_face)
            
        elif step == "D'":
            [up_face, right_face, front_face, down_face, left_face, back_face] = down_ccw(video,  up_face, right_face, front_face, down_face, left_face, back_face)
            
        elif step == "D2":
            [up_face, right_face, front_face, down_face, left_face, back_face] = down_cw(video,  up_face, right_face, front_face, down_face, left_face, back_face)
            [up_face, right_face, front_face, down_face, left_face, back_face] = down_cw(video,  up_face, right_face, front_face, down_face, left_face, back_face)
    
    cv.imshow("Output Image", frame)
    key_pressed = cv.waitKey(1) & 0xFF
    if key_pressed == 27 or key_pressed == ord('q'):
        break