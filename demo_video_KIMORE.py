import copy
import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
import math
import argparse
import skvideo.io

# openpose setup
from src import model
from src import util
from src.body import Body
from src.hand import Hand

# writing video with ffmpeg because cv2 writer failed
# https://stackoverflow.com/questions/61036822/opencv-videowriter-produces-cant-find-starting-number-error
import ffmpeg

COCO_UPPER_BODY_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11]
VALID_BODY_JOINTS = COCO_UPPER_BODY_JOINTS
body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')

TEST_SINGLE_VIDEO = False
VIDEO_NAME = "Blur_rgb040716_112230.mp4"


def is_data_valid(subset):
    if (len(subset) == 0):
        return False
    num_missing_joints = (subset[0][:-2] == -1).sum()
    if (num_missing_joints >= 8 ):
        print("Missing %i points".format(num_missing_joints))
        return False

    for i in VALID_BODY_JOINTS:
        if (i not in subset[0][:-2]):
            print("missing important data index:" + str(i))
            return False
    return True

def get_18_body_points(candidate, subset):
    body_points = {}
    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            # Only process index 0 - 13 ( total 14 points)
            if index > 13:
                continue
            x, y = candidate[index][0:2]
            body_points[index] = [x,y]
    return body_points

def list_to_file(file, body_points):
    separator = ","
    counter = 1
    for i in VALID_BODY_JOINTS:
        x, y = body_points[i]
        if (counter == len(VALID_BODY_JOINTS)):
            file.write(str(x) + separator + str(y))
        else:
            file.write(str(x) + separator + str(y) + separator)
        counter += 1
    file.write("\n")

def process_frame(txt_file, frame, body=True, hands=False):
    canvas = copy.deepcopy(frame)
    if body:
        candidate, subset = body_estimation(frame)
        if (is_data_valid(subset)):
            body_points = get_18_body_points(candidate, subset)
            list_to_file(txt_file, body_points)
        else:
            canvas = copy.deepcopy(frame)
            canvas = util.draw_bodypose(canvas, candidate, subset)

            # plt.imshow(canvas[:, :, [2, 1, 0]])
            # plt.axis('off')
            # plt.show()

            print(subset)

    if hands:
        hands_list = util.handDetect(candidate, subset, frame)
        all_hand_peaks = []
        for x, y, w, is_left in hands_list:
            peaks = hand_estimation(frame[y:y+w, x:x+w, :])
            peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
            peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
            all_hand_peaks.append(peaks)
        canvas = util.draw_handpose(canvas, all_hand_peaks)
    return canvas

def to8(img):
    return (img/256).astype('uint8')

class Writer():
    def __init__(self, output_file, input_fps, input_framesize, gray=False):
        if os.path.exists(output_file):
            os.remove(output_file)
        self.ff_proc = (
            ffmpeg.input('pipe:',
                   format='rawvideo',
                   pix_fmt='gray' if gray else 'rgb24',
                   s='%sx%s'%(input_framesize[1],input_framesize[0]))
            .filter('fps', fps=input_fps, round='up')
            .output(output_file, pix_fmt='yuv420p')
            .run_async(pipe_stdin=True)
        )

    def __call__(self, frame):
        self.ff_proc.stdin.write(frame.tobytes())

    def close(self):
        self.ff_proc.stdin.close()
        self.ff_proc.wait()


def test_single_video(fps):
    video_file = "/Users/Clara_1/Google Drive/KiMoRe/RGB/GPP/Stroke/S_ID6/Es1/rgb/" + VIDEO_NAME
    new_name = os.path.join(*(video_file.split("/")[6:]))
    new_name = new_name.replace("/", "_").split(".")[0]
    output_txt_file = "my_data/tmp/" + new_name + ".txt"
    print(new_name)

    # Create a txt file to write body joints into
    txt_file = open(output_txt_file, "w+")

    X = skvideo.io.vread(video_file, outputdict={'-r': fps})  # (frames, height, width, channel)
    frame_count = 0
    for i in range(X.shape[0]):
        frame = X[i]
        posed_frame = process_frame(txt_file, frame, body=True, hands=False)
        frame_count += 1
        print(frame_count)

    txt_file.close()


def main():
    # Configs
    video_data_root = "/Users/Clara_1/Google Drive/KiMoRe/RGB"
    exercise_type = "Es1"
    include_body = True
    include_hand = False
    fps = "10"     # needs to be a string

    num_video = 0

    if (TEST_SINGLE_VIDEO):
        test_single_video(fps)
        return

    for video_file in glob.glob(video_data_root + "/**/*.mp4", recursive=True):
        if exercise_type not in video_file:
            continue


        new_name = os.path.join(*(video_file.split("/")[6:]))
        new_name = new_name.replace("/", "_").split(".")[0]
        output_txt_file = "my_data/output_3_12/" + new_name + ".txt"
        print(new_name)

        # Create a txt file to write body joints into
        txt_file = open(output_txt_file, "w+")

        X = skvideo.io.vread(video_file, outputdict={'-r': fps})  # (frames, height, width, channel)
        frame_count = 0
        for i in range(X.shape[0]):
            frame = X[i]
            posed_frame = process_frame(txt_file, frame, body=include_body, hands=include_hand)
            frame_count += 1
            print(frame_count)

        txt_file.close()
        num_video += 1
        #
        # # open specified video
        # cap = cv2.VideoCapture(video_file)
        #
        # # pull video file info
        # # don't know why this is how it's defined https://stackoverflow.com/questions/52068277/change-frame-rate-in-opencv-3-4-2
        # input_fps = cap.get(5)
        #
        # # define a writer object to write to a movidified file
        # assert len(video_file.split(".")) == 2, \
        #     "file/dir names must not contain extra ."
        # output_file = video_file.split(".")[0] + ".processed.avi"
        #
        # # writer = None
        # counter = 0
        # while (cap.isOpened()):
        #
        #     frameId = cap.get(1)  # current frame number
        #     print('id = ' + str(frameId))
        #     print('est = ' + str((math.floor(input_fps)//fps)))
        #     if (frameId % (math.floor(input_fps)//fps) != 0):
        #         continue
        #
        #     ret, frame = cap.read()
        #     if frame is None:
        #         break
        #
        #     # if writer is None:
        #     #     input_framesize = frame.shape[:2]
        #     #     writer = Writer(output_file, fps, input_framesize)
        #
        #     posed_frame = process_frame(txt_file, frame, body=include_body, hands=include_hand)
        #
        #     # cv2.imshow('frame', posed_frame)
        #
        #     # write the frame
        #     # writer(posed_frame)
        #
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        #
        #     counter += 1
        #     print(counter)
        #
        # cap.release()
        # # writer.close()
        # cv2.destroyAllWindows()
        # txt_file.close()


    print("Exercise type: " + exercise_type + " has " + num_video + " videos")

if __name__ == "__main__":
    # execute only if run as a script
    main()
