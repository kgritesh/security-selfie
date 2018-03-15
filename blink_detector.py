# -*- coding: utf-8 -*-
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import math


class BlinkDetector:
    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 3

    def __init__(self, shape_predictor_model, fl):
        self.video_file = fl
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(shape_predictor_model)

        self.counter = 0
        self.blinks = []


    @classmethod
    def eye_aspect_ratio(cls, eye):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])

        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])

        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)

        # return the eye aspect ratio
        return ear

    def start(self):
        vs = cv2.VideoCapture(self.video_file)
        no_frames = vs.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = vs.get(cv2.CAP_PROP_FPS)
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        i = 0
        blink_frames = []
        while True:
            (grabbed, frame) = vs.read()
            if not grabbed:
                break
            i += 1
            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detect faces in the grayscale frame
            rects = self.detector(gray, 0)

            # loop over the face detections
            for rect in rects:
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # extract the left and right eye coordinates, then use the
                # coordinates to compute the eye aspect ratio for both eyes
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = self.eye_aspect_ratio(leftEye)
                rightEAR = self.eye_aspect_ratio(rightEye)

                # average the eye aspect ratio together for both eyes
                ear = (leftEAR + rightEAR) / 2.0

                if ear < self.EYE_AR_THRESH:
                    self.counter += 1

                # otherwise, the eye aspect ratio is not below the blink
                # threshold
                else:
                    # if the eyes were closed for a sufficient number of
                    # then increment the total number of blinks
                    if self.counter >= self.EYE_AR_CONSEC_FRAMES:
                        blink_frames.append(i)

                    # reset the eye frame counter
                    self.counter = 0

        fps = i / 7
        print('Total Frames: {}, FPS: {}'.format(i, fps))
        self.blinks = list(map(lambda fr: math.floor(fr / fps), blink_frames))









