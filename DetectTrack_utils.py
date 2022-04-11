import cv2
import numpy as np


def loadModel():
    modelConfig = 'yolov3.cfg'
    modelWeights = 'yolov3.weights'
    net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
    return net


def classIndex(className):
    classesFile = 'coco.names'
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    classes = np.asarray(classes)
    return np.squeeze(np.argwhere(classes == className))


def detectObject(net, class_index, frame, inSize=(416, 416), thresholds=None):
    """
    :param net: dnn object
    :param class_index: class id to be detected
    :param frame: image on which to perform detection
    :param inSize: YOLOv3 input image size
    :param thresholds: None or tuple of object, bbox, and NMS thresholds
    :return:
    """
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1 / 255, size=inSize,
                                 mean=[0, 0, 0], swapRB=True, crop=True)

    net.setInput(blob)
    # extract outputs of all the different output layers of the YOLO model
    unconnectedLayerNames = net.getUnconnectedOutLayersNames()
    outs = net.forward(unconnectedLayerNames)

    # here, blob includes only one "sample image"
    frame = blob2Image(blob)

    if thresholds:
        objectnessThresh, confidenceThresh, nmsThresh = thresholds
        bbox, confidence = postprocess(frame.shape,
                                       outs,
                                       class_index,
                                       objectnessThresh,
                                       confidenceThresh,
                                       nmsThresh)
    else:
        bbox, confidence = postprocess(frame.shape, outs, class_index)

    return bbox, confidence


def blob2Image(blob):
    """
        use "blob2Image" for the case that blob changed the aspect ratio
        of the original image, so blob coordinates are different from the
        original coordinates
    """
    im = np.squeeze(blob)  # (channels, height, width)
    im = np.transpose(im, [1, 2, 0])  # (height, width, channels)
    im = cv2.cvtColor(np.uint8(255 * im), cv2.COLOR_RGB2BGR)
    return im


def postprocess(frameShape, outs, class_index, objectnessThresh=.5,
                confidenceThresh=.5, nmsThresh=.4):
    """
    given a set of bounding boxes and defined thresholds, extract the highest
    probability bounding box for the given object class

    use Non-Max Suppression

    :param frame: image to detect object in
    :param outs: outcome of forward-pass
    :param class_index: object class index to be detected
    :return: first bbox
    """

    H, W, _ = frameShape

    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            if detection[4] > objectnessThresh and \
                    detection[5 + class_index] > confidenceThresh:
                # extract bounding box position and rescale to frame
                # coordinates. YOLO returns center of bbox while
                # OpenCV convention is top-left corner + width and height
                center_x = int(detection[0] * W)
                center_y = int(detection[1] * H)
                width = int(detection[2] * W)
                height = int(detection[3] * H)

                left = int(center_x - width / 2)
                top = int(center_y - height / 2)

                boxes.append((left, top, width, height))
                confidences.append(float(detection[5 + class_index]))

    # perform Non-Max Suppression to avoid multiple detection of the same object
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidenceThresh, nmsThresh)
    if len(indices):
        bbox = boxes[indices[0][0]]
        confidence = confidences[indices[0][0]]
    else:
        bbox = None
        confidence = None

    return bbox, confidence


def drawBox(frame, source, args=None):
    """
    draw bounding box unique to source

    :param frame: image to draw box on
    :param source: source of detection: 'TRACKING'/'DETECTION'/'NONE'
    :param args: depending on source, either (bbox, confidence) for detection
    or (bbox, TRACKER) for tracking
    :return: frame ready for display
    """

    if source == 'NONE':
        label = '**OBJECT LOST**'
        label_size, base_line = cv2.getTextSize(label,
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                fontScale=1,
                                                thickness=2
                                                )
        cv2.rectangle(frame, (0, 0),
                      (round(1.2 * label_size[0]),
                      round(1.2 * (label_size[1] + base_line))),
                      (128, 128, 128), cv2.FILLED
                      )
        cv2.putText(frame, label, (round(0.1 * label_size[0]),
                    round(1.1 * (label_size[1] + base_line))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 0, 255),
                    thickness=2
                    )

    if source == 'DETECTION':
        bbox, confidence = args
        left, top, width, height = bbox
        topLeft = (left, top)
        bottomRight = (left + width, top + height)

        cv2.rectangle(frame, topLeft, bottomRight, (255, 0, 0), 3)
        label = f'{round(confidence, 4)}'

        label_size, base_line = cv2.getTextSize(label,
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                .5, 1
                                                )
        top = max(top, label_size[1])
        cv2.rectangle(frame, (left, top - round(1.5 * label_size[1])),
                      (left + round(1.5 * label_size[0]), top + base_line),
                      (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (0, 0, 0), 1
                    )

    if source == 'TRACKING':
        bbox, tracker = args
        left, top, width, height = bbox
        topLeft = (left, top)
        bottomRight = (left + width, top + height)

        cv2.rectangle(frame, topLeft, bottomRight, (0, 255, 0), 3)
        label = f'tracking ({tracker})'

        label_size, base_line = cv2.getTextSize(label,
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                .5, 1
                                                )
        top = max(top, label_size[1])
        cv2.rectangle(frame, (left, top - round(1.5 * label_size[1])),
                      (left + round(1.5 * label_size[0]), top + base_line),
                      (255, 255, 255), cv2.FILLED
                      )
        cv2.putText(frame, label, (left, top),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (0, 0, 0), 1
                    )

    return frame


def initializeTracker(tracker_type):
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    elif tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()
    elif tracker_type == "MOSSE":
        tracker = cv2.TrackerMOSSE_create()
    else:
        tracker = None
        print('Incorrect or unsupported tracker name')
        print('Use TLD, KCF, CSRT or MOSSE')

    return tracker


def bbox_hist(frame, bbox):
    """
    this function returns histogram performed over the extracted bounding box
    based on Hue (color) component

    :param frame: frame image
    :param bbox: bounding box tuple
    :return: normalized histogram of bbox
    """

    # first handle out of frame indices in bbox
    H, W, _ = frame.shape
    left, top, width, height = bbox
    if left < 0:
        width += left
        left = 0
    if left + width > W-1:
        width = W-1 - left

    if top < 0:
        height += top
        top = 0
    if top + height > H:
        height = H-1 - top
    bbox = (left, top, width, height)


    windowHSV = cv2.cvtColor(
                    frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2], :],
                    cv2.COLOR_BGR2HSV
                    )

    # Hue can have 0 to 180 value, thus 181 bins.
    bins = [b for b in range(181)]
    bbox_hist, bins = np.histogram(windowHSV[...,0], bins=bins)
    return bbox_hist / np.sum(bbox_hist)


def calcHistDist(hist1, hist2):
    """
    using "histogram intersection" for distance
    a better method to deal with occlusions
    :param hist1:
    :param hist2:
    :return: distance between the two
    """
    # L2 distance
    # return (1/180) * np.sum(np.power((hist1-hist2), 2))

    # Intersection
    return np.sum(np.where(hist1<=hist2, hist1, hist2))