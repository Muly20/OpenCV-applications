import numpy as np
import cv2

def blemishRemoval(action, y, x, flags, userdata):
    """
    callback function for mouse left button click
    calls findBestSlice and uses seamless cloning to replace patch
    with best match
    """
    global image, r
    if action==cv2.EVENT_LBUTTONDOWN:
        center = (x,y)
        # return if search window is out of frame
        if x<3*r or x>image.shape[0]-3*r-1: return
        if y<3*r or y>image.shape[1]-3*r-1: return

        slice_i, slice_j = findBestSlice(center, r)
        patch = image[slice_i-r:slice_i+r+1, slice_j-r:slice_j+r+1, :]
        # image = cv2.circle(image, (y,x), r, (0,255,0), thickness=1)
        # image = cv2.circle(image, (slice_j, slice_i), r, (255,0,0), thickness=1)

        mask = 255*np.ones(patch.shape, patch.dtype)
        image = cv2.seamlessClone(patch, image, mask, (y,x), cv2.NORMAL_CLONE)
        cv2.imshow('blemish removal', image)

def findBestSlice(center, r):
    """
    returns center of best match (radius 'r') for path in 8 patches around ROI.
    Use both minimum border distance for similarity measure and minimum
    gradient (using Sobel kernel) for smoothness measure
    """
    global image
    x, y = center

    # flatten ROI border values for later comparison
    border_y = np.array((image[x-r:x+r+1, y-r,:],
                         image[x-r:x+r+1, y+r,:])
                        ).reshape(-1,3)
    border_x = np.array((image[x-r, y-r:y+r+1,:],
                         image[x+r, y-r:y+r+1,:])
                        ).reshape(-1,3)
    border_orig = np.array((border_x, border_y)).reshape(-1,3)

    # arrays for storing grad and border difference values between ROI and
    # potential displacing patch
    grad_array = (-1) * np.ones((3,3))
    border_array = (-1) * np.ones((3,3))

    # run over 8 closest patches
    for i in range(-1,2):
        for j in range(-1,2):
            if not (i==0 and j==0):
                x_loc = x+i*2*r
                y_loc = y+j*2*r

                slice = image[x_loc-r:x_loc+r+1, y_loc-r:y_loc+r+1,:]

                # determine gradients of the (i,j) slice
                grad_x = cv2.Sobel(slice, cv2.CV_32F, 1, 0, ksize=3)
                grad_x = np.mean(np.power(grad_x, 2))
                grad_y = cv2.Sobel(slice, cv2.CV_32F, 0, 1, ksize=3)
                grad_y = np.mean(np.power(grad_y, 2))

                # flatten slice's borders and calculate L2 distance with
                # original patch border
                slice_border_y = np.array((slice[:, 0,:],
                                           slice[:, -1,:])
                                          ).reshape(-1,3)
                slice_border_x = np.array((slice[0, :,:],
                                           slice[-1, :,:])
                                          ).reshape(-1,3)
                border_slice = np.array((slice_border_x, slice_border_y)
                                        ).reshape(-1,3)
                border_diff = np.mean(np.power(border_slice - border_orig, 2))

                # store gradients and border diff
                grad_array[i+1, j+1] = grad_x+grad_y
                border_array[i+1, j+1] = border_diff


    # take out the "center" (original patch)
    border_array = border_array.reshape(-1)[np.array([0,1,2,3,5,6,7,8])]
    grad_array = grad_array.reshape(-1)[np.array([0,1,2,3,5,6,7,8])]

    # rescale metrics
    border_min = np.min(border_array)
    border_max = np.max(border_array)
    border_array = (border_array - border_min) / (border_max - border_min)

    grad_min = np.min(grad_array)
    grad_max = np.max(grad_array)
    grad_array = (grad_array - grad_min) / (grad_max - grad_min)

    # extract patch index/location based on similarity criteria
    d = np.argmin(border_array + grad_array)
    # skip middle slot (original patch)
    if d>=4:
        d += 1

    di = d//3-1
    dj = d % 3-1

    return (x+di*2*r, y+dj*2*r)

"""
main code

load image and allows for blemish removal through mouse-click
click 'c' to clear changes (back to original image)
click 's' to save image to file ('blemish_save.png')
click 'ESC' to exit
"""
r = 15
image = cv2.imread('blemish.png', cv2.IMREAD_UNCHANGED)
imageCopy = image.copy()
cv2.namedWindow('blemish removal')

# set mouse callback
cv2.setMouseCallback('blemish removal', blemishRemoval)
k=0
# hit 'ESC' to exit
while(k!=27):
    cv2.imshow('blemish removal', image)
    k = cv2.waitKey(20) & 0xFF
    if k==99:
        # clear changes
        image = imageCopy.copy()
    if k==115:
        cv2.imwrite('belmish_save.png', image)

cv2.destroyAllWindows()