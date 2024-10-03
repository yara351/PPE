import asone
from asone import ASOne
import cv2
import random
import time
import numpy as np
from tracker import *

sizeLogo = 100
vestGreen = cv2.imread("static/files/greenVest.png")
vestRed = cv2.imread("static/files/redVest.png")
helmetGreen = cv2.imread("static/files/greenHelmet.png")
helmetRed = cv2.imread("static/files/redHelmet.png")
maskGreen = cv2.imread("static/files/greenMask.png")
maskRed = cv2.imread("static/files/redMask.png")


vestGreen = cv2.resize(vestGreen, (sizeLogo, sizeLogo))
vestRed = cv2.resize(vestRed, (sizeLogo, sizeLogo))
helmetGreen = cv2.resize(helmetGreen, (sizeLogo, sizeLogo))
helmetRed = cv2.resize(helmetRed, (sizeLogo, sizeLogo))
maskGreen = cv2.resize(maskGreen, (sizeLogo, sizeLogo))
maskRed = cv2.resize(maskRed, (sizeLogo, sizeLogo))


img2grayH = cv2.cvtColor(helmetRed, cv2.COLOR_BGR2GRAY)
retH, maskH = cv2.threshold(img2grayH, 1, 255, cv2.THRESH_BINARY)
img2grayV = cv2.cvtColor(vestRed, cv2.COLOR_BGR2GRAY)
retV, maskV = cv2.threshold(img2grayV, 1, 255, cv2.THRESH_BINARY)
img2grayM = cv2.cvtColor(maskRed, cv2.COLOR_BGR2GRAY)
retV, maskM = cv2.threshold(img2grayM, 1, 255, cv2.THRESH_BINARY)


def plot_one_boxCustom(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    try:
        if "head_whelmet" in label:
            roihelmet = img[c1[1]+70-sizeLogo:c1[1]+70, c1[0]-sizeLogo:c1[0]]
            roihelmet[np.where(maskH)] = 0
            roihelmet += helmetGreen
        else:
            roihelmet = img[c1[1]+70-sizeLogo:c1[1]+70, c1[0]-sizeLogo:c1[0]]
            roihelmet[np.where(maskH)] = 0
            roihelmet += helmetRed
    except:
        pass
    try:
        if "face_wmask" in label:
            roimask = img[c1[1]+110+70-sizeLogo:c1[1] +
                          110+70, c1[0]-sizeLogo:c1[0]]
            roimask[np.where(maskM)] = 0
            roimask += maskGreen
        else:
            roimask = img[c1[1]+110+70-sizeLogo:c1[1] +
                          110+70, c1[0]-sizeLogo:c1[0]]
            roimask[np.where(maskM)] = 0
            roimask += maskRed
    except:
        pass
    try:
        if "vest" in label:
            roivest = img[c1[1]+210+70-sizeLogo:c1[1] +
                          210+70, c1[0]-sizeLogo:c1[0]]
            roivest[np.where(maskV)] = 0
            roivest += vestGreen
        else:
            roivest = img[c1[1]+210+70-sizeLogo:c1[1] +
                          210+70, c1[0]-sizeLogo:c1[0]]
            roivest[np.where(maskV)] = 0
            roivest += vestRed
    except:
        pass


def video_detection(path_x='', conf_=0.25):

    names = ['face_nomask', 'face_wmask', 'hand_noglove',
             'hand_wglove', 'head_nohelmet', 'head_whelmet', 'person', 'vest']
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    filter_classes = None

    if filter_classes:
        filter_classes = [filter_classes]
    frameCounter = 0
    dt_obj = ASOne(
        tracker=asone.BYTETRACK,
        detector=asone.YOLOV8N_PYTORCH,
        weights="best.pt",
        use_cuda=False
    )
    start_time = time.time()
    total_detections = 0

    video_path = path_x

    video = cv2.VideoCapture(video_path)
    nframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker = EuclideanDistTracker()

    for j in range(nframes):
        safePersons = []
        detectionsCount = []
        ret, img0 = video.read()
        if ret:
            detected = dt_obj.detect(source=img0, conf_thres=0.25, iou_thres=0.45,
                                     filter_classes=filter_classes)
            output_detected = detected[0]
            detectionTracker = []
            equipmentList = []
            for results in output_detected:
                box = [results[0], results[1], results[2], results[3]]
                label = f'{names[int(results[5])]} {results[4]:.2f}'

                if names[int(results[5])].strip() == 'person':
                    detectionTracker.append(
                        [box[0], box[1], box[2], box[3], label])
                else:
                    equipmentList.append(
                        [box[0], box[1], box[2], box[3], label])
            boxes_ids = tracker.update([detectionTracker, equipmentList])
            for box_id in boxes_ids:
                x, y, w, h, id, label_ = box_id
                labelPlot = label_
                if id in detectionsCount:
                    pass
                else:
                    detectionsCount.append(id)

                if "vest" in label_ and "head_whelmet" in label_ and "face_wmask" in label_:
                    safePersons.append(id)
                if frameCounter % 17 == 0:
                    labelPlot = label_
                plot_one_boxCustom([int(x), int(y), int(w), int(
                    h)], img0, label=labelPlot, color=colors[int(0)], line_thickness=3)
                frameCounter += 1
                yield img0, len(list(set(detectionsCount))), len(list(set(safePersons)))

    video.release()
