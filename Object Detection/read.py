import cv2
import numpy as np

#green = [36, 202, 59, 71, 255, 255]
#yellow = [18, 0, 196, 36, 255, 255]
blue = [95, 120, 20, 112, 255, 255]
red = [170, 120, 50, 180, 255, 255]

color=blue

# Initialize Video Feed
cap1 = cv2.VideoCapture('Videos/Red.mp4')
cap2 = cv2.VideoCapture('Videos/Blue.mp4')
cap3 = cv2.VideoCapture(0)

while(True):

    # Take each frame
    _, frame1 = cap1.read()
    _, frame2 = cap2.read()
    _, frame3 = cap3.read()
    
    # stacking up all images together
    stacked = np.hstack((frame3,frame3))

    frameHSV = cv2.cvtColor(stacked, cv2.COLOR_BGR2HSV)
    
    # HSV values to define a colour range we want to create a mask from.
    colorLow = np.array(color[0:3])
    colorHigh = np.array(color[3:6])
    mask = cv2.inRange(frameHSV, colorLow, colorHigh)

    contours, h = cv2.findContours(mask, 1, 2)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:]

    for cnt in contours:
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)

        # only proceed if the radius meets a minimum size
        if radius > 5 and cv2.contourArea(cnt)/(np.pi*radius**2) > 0.75:
            M = cv2.moments(cnt)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            cv2.drawContours(stacked, cnt, -1, (0,255,0), 3 )

            middleX = int(x)
            middleY = int(y)
            percentX = (middleX - stacked.shape[0])
            percentY = (middleY - stacked.shape[1])

            # draw the circle and centroid on the frame
            cv2.circle(stacked, (middleX,middleY), int(radius),(0, 0, 255), 2)
            cv2.circle(stacked, (middleX,middleY), 5, (0, 0, 255), -1)

    # Show final output image
    cv2.imshow('Result', stacked)
    
    if cv2.waitKey(20) & 0xFF==ord('q'):
        break

    
cv2.destroyAllWindows()
cap1.release()
cap2.release()