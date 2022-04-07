import cv2
import numpy as np

def changeBackground(frame, background, args, kernel):
    global bg_color_loc

    # do nothing until user select color
    if not bg_color_loc: return frame
    y, x = bg_color_loc

    delta, maxS, minS, maxV, minV, ksize = args
    # extract background
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(np.int32(frameHSV))
    base_color = H[x, y]

    # identify background pixels using trackbar values for HSV
    idx_2d = np.argwhere((np.abs(H - base_color) <= delta) & ((V <= maxV) & (V >= minV)) & ((S <= maxS) & (S >= minS)))
    idx_i, idx_j = np.hsplit(idx_2d, 2)

    # locate boundaries / edges
    grad = cv2.Sobel(np.squeeze(frame[..., 1]), cv2.CV_32F, 1, 1)
    cv2.normalize(grad, grad, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    grad = np.uint8(255 * grad)
    grad = cv2.dilate(grad, kernel, iterations=2)

    # applying background
    frame[idx_i, idx_j, :] = background[idx_i, idx_j, :]

    # dealing with green on the boundaries by applying medianBlur on them
    if ksize:
        # kernel has to be un-even
        if ksize % 2 == 0: ksize += 1
        # blur contours and mix with updated-background frame
        _, mask = cv2.threshold(grad, 150, 255, cv2.THRESH_BINARY)
        blur = cv2.medianBlur(frame, ksize)
        blur[mask == 0, :] = [0, 0, 0]
        frame[mask != 0, :] = [0, 0, 0]
        frame = frame + blur

    return frame


def onChange(*args):
    pass


def chooseColor(action, y, x, flags, userdata):
    global bg_color_loc
    if action == cv2.EVENT_LBUTTONDOWN:
        bg_color_loc = (y, x)


if __name__ == '__main__':

    background = cv2.imread('bluesky.jpg')

    cap = cv2.VideoCapture('video_greenbackground.mp4')
    if not cap.isOpened(): print("Cannot open file")

    # adjust background to video frame size
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/2)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/2)
    background_resized = cv2.resize(background, (W, H))

    # define trackbars
    panel = np.zeros((200, 1000), np.uint8)
    cv2.namedWindow('Control Panel')
    cv2.createTrackbar('Background Color Tolerance', 'Control Panel', 10, 50, onChange)
    cv2.createTrackbar('Max Saturation', 'Control Panel', 255, 255, onChange)
    cv2.createTrackbar('Min Saturation', 'Control Panel', 60, 255, onChange)
    cv2.createTrackbar('Max Brightness', 'Control Panel', 255, 255, onChange)
    cv2.createTrackbar('Min Brightness', 'Control Panel', 50, 255, onChange)
    cv2.createTrackbar('Boundary Blur Size', 'Control Panel', 5, 13, onChange)
    cv2.imshow('Control Panel', panel)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    cv2.namedWindow('video')
    cv2.setMouseCallback('video', chooseColor)
    bg_color_loc = None

    k = 0
    while (cap.isOpened() and k != 27):
        ret, frame = cap.read()

        if not ret:
            print("Cannot read frame")
            exit()
        frame = cv2.resize(frame, (W, H))

        # extract arguments from panel
        delta = cv2.getTrackbarPos('Background Color Tolerance', 'Control Panel')
        maxS = cv2.getTrackbarPos('Max Saturation', 'Control Panel')
        minS = cv2.getTrackbarPos('Min Saturation', 'Control Panel')
        maxV = cv2.getTrackbarPos('Max Brightness', 'Control Panel')
        minV = cv2.getTrackbarPos('Min Brightness', 'Control Panel')
        ksize = cv2.getTrackbarPos('Boundary Blur Size', 'Control Panel')
        panel_args = (delta, maxS, minS, maxV, minV, ksize)

        # get updated frame
        newframe = changeBackground(frame, background_resized, panel_args, kernel)
        cv2.imshow('video', newframe)
        k = cv2.waitKey(20) & 0xFF

    cap.release()
    cv2.destroyAllWindows()