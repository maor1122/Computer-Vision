# ======= imports
import matplotlib.pyplot as plt
import numpy as np
import cv2

# ======= constants
figsize = (10, 10)
prevImg = cv2.imread("prevImage.jpg")
prevRGB = cv2.cvtColor(prevImg, cv2.COLOR_BGR2RGB)
gray_prev = cv2.cvtColor(prevRGB, cv2.COLOR_RGB2GRAY)

newImg = cv2.imread("newImage.jpg")
newImgRGB = cv2.cvtColor(newImg, cv2.COLOR_BGR2RGB)

# === template image keypoint and descriptors
feature_extractor = cv2.SIFT_create()
kp_prev, desc_prev = feature_extractor.detectAndCompute(gray_prev, None) # find the keypoints and descriptors with chosen feature_extractor


# ===== video input, output and metadata
videoName = "inputVid.mp4" # Insert video name here.
cap = cv2.VideoCapture(videoName)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_size = (frame_width,frame_height)
output = cv2.VideoWriter('outputVid.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20, frame_size) # Initialize video writer object


def changePic(frame):
    global newImgRGB
    # Find keypoints and descriptors with the current frame (SIFT feature detection)
    curRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_cur = cv2.cvtColor(curRGB, cv2.COLOR_RGB2GRAY)
    kp_cur, desc_cur = feature_extractor.detectAndCompute(gray_cur, None)

    # Take only unique features
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_prev, desc_cur, k=2)
    # Apply ratio test
    good_and_second_good_match_list = []
    for m in matches:
        if m[0].distance/m[1].distance < 0.5:
            good_and_second_good_match_list.append(m)
    good_match_arr = np.asarray(good_and_second_good_match_list)[:,0]
    
    # Finding Homography between images
    good_kp_prev = np.array([kp_prev[m.queryIdx].pt for m in good_match_arr])
    good_kp_cur = np.array([kp_cur[m.trainIdx].pt for m in good_match_arr])
    H, masked = cv2.findHomography(good_kp_prev, good_kp_cur, cv2.RANSAC, 5.0)

    # Resizing to be able to overlay images 
    width = int(prevRGB.shape[1])
    height = int(prevRGB.shape[0])
    newImgRGB = cv2.resize(newImgRGB, (width, height))

    # Overlay images
    h1, w1 = curRGB.shape[:2]
    h2, w2 = newImgRGB.shape[:2]
    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])  # translate

    result = cv2.warpPerspective(newImgRGB, Ht@H, (xmax-xmin, ymax-ymin))      
    
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    curRGB[result>0] = result[result>0]

    return curRGB

# ========== run on all frames
while(cap.isOpened()):    
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Checks if the video is over
    if frame is None:
        break
    
    # Display the resulting frame
    if frame.size != 0:
        res = changePic(frame)
        cv2.imshow('Frame', res)
        # =========== plot and save frame
        output.write(res) # Write the frame to the output files
        

    # *OPTIONAL*, use ESC to exit the video.
    k = cv2.waitKey(1)
    if k==27:    # Esc key to stop
        break

# ======== end all
# When everything done, release the capture
cap.release()
output.release()
cv2.destroyAllWindows()

