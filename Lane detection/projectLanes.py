from matplotlib import pyplot as plt
import cv2
import numpy as np


def findLanes(frame):
    imageShape = frame.shape
    offsetTop = int(imageShape[0]*5/8)
    offsetLeft = int(imageShape[1]*1/6)
    im = frame.copy()

    # Cropping the image in center.
    im = im[offsetTop:, offsetLeft:5*offsetLeft]

    # Bluרring/removing noise to reduce edges from canny.
    medianPic = cv2.medianBlur(im, 7)
    medianPic = cv2.medianBlur(medianPic, 3)

    # Detecting edges. 
    im_canny = getCannyPic(medianPic)

    # Drawing the two lines and returning lane change.
    resVal = drawLines(im_canny, im)

    # Merging the picture with the original image
    frame[offsetTop:,offsetLeft:5*offsetLeft]=im
        
    return [frame, resVal]


lastCorrectLeftV = [0, 0]
lastCorrectRightV = [0, 0]
lastCorrectLeftB = 1000
lastCorrectRightB = 1000
counter = 0
lastLaneChange = 0


def getCannyPic(im):
    imGray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) # Convert image to black and while.
    imGray[imGray < 190]=0 # Threshold the image by light level
    return cv2.Canny(imGray, 150, 200) # Get image edges


def drawLines(cannyPic, croppedFrame):
    global lastCorrectLeftV
    global lastCorrectRightV
    global lastCorrectLeftB
    global lastCorrectRightB
    lines = cv2.HoughLines(cannyPic, 1, np.pi / 180, 50, 100, 1, 100) #Get detected image lines.
    
    res = 0
    rightV, leftV = [0,0],[0,0]

    minLeftB = 1000 
    minRightB = 1000

    if lines is not None:
        # print(lines.shape)
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            if ((x2-x1)==0):
                continue

            m=(y2-y1)/(x2-x1)
            b=y1-m*x1
            threshold = 0.5
            
            if (m > threshold):
                if (b < minLeftB):
                    minLeftB = b
                    if (lastCorrectLeftB != 1000 and minLeftB != 1000):
                            if (lastCorrectLeftB < 2.5*(minLeftB)):
                                res = 1
                    leftV = [(x1,y1),(x2,y2)]
                    lastCorrectLeftB = b

            if (m < (-threshold)):
                 if (b < minRightB):
                    minRightB = b
                    if (lastCorrectRightB != 1000 and minRightB != 1000):
                            if (lastCorrectRightB > 2.5*(minRightB)):
                                res = 2
                    rightV = [(x1,y1),(x2,y2)]
                    lastCorrectRightB = b
    
    if not(leftV[0] or leftV[1]):
        leftV = lastCorrectLeftV

    if not(rightV[0] or rightV[1]):
        rightV = lastCorrectRightV

    
    if (leftV[0] or leftV[1]):
        cv2.line(croppedFrame,leftV[0],leftV[1],(0,0,255),2)
        lastCorrectRightV = rightV
    
    if (rightV[0] or rightV[1]):
        cv2.line(croppedFrame,rightV[0],rightV[1],(0,0,255),2)
        lastCorrectLeftV = leftV
    return res

def drawLaneChange(lastLaneChange, frame):
    #=== font vars
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.8
    fontColor = (0,0,0)
    lineType = 2
    
    pos = [int(frame.shape[1]/2-70),int(frame.shape[0]*1/5)]
    # right
    if (lastLaneChange == 1):
        cv2.putText(frame, "Moving Right", (pos[0], pos[1]), font, fontScale, fontColor, lineType)
    # left
    if (lastLaneChange == 2):
        cv2.putText(frame, "Moving Left", (pos[0], pos[1]), font, fontScale, fontColor, lineType)

videoName = "roadtrip.mp4" # Insert video name here.
cap = cv2.VideoCapture(videoName)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_size = (frame_width,frame_height)
# Initialize video writer object
output = cv2.VideoWriter('roadtripOutput.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20, frame_size)

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Checks if the video is over
    if frame is None:
        break

    # Display the resulting frame
    if frame.size != 0:
        res, laneChange = findLanes(frame)
    
        if (laneChange != 0):
            counter = 45
            lastLaneChange = laneChange
        if (counter != 0):
            drawLaneChange(lastLaneChange, res)
            counter -= 1

        cv2.imshow('Frame', res)
        # Write the frame to the output files
        output.write(frame)
    
    # *OPTIONAL*, use ESC to exit the video.
    k = cv2.waitKey(1)
    if k==27:    # Esc key to stop
        break
    
# When everything done, release the capture
cap.release()
output.release()
cv2.destroyAllWindows()