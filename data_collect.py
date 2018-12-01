import math

import cv2
import numpy as np
import random
from collections import deque


def checkletter(img):

    cv2.imwrite("x" + ".png", img)
    
    cv2.imshow('croppws', img)
    cv2.waitKey(1)

    

if __name__ == '__main__':

    cap = cv2.VideoCapture(0)
    # To keep track of all point where object visited
    center_points = deque()
    counter=0

    while True:
        counter+=1

        # Read and flip frame
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # Blur the frame a little
        #blur_frame = cv2.GaussianBlur(frame, (7, 7), 0)

        # Convert from BGR to HSV color format
        # hsv = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV)

        # Define lower and upper range of hsv color to detect. Blue here
        #lower_blue = np.array([100, 50, 50])
        #upper_blue = np.array([140, 255, 255])

        # Define lower and upper range of hsv color to detect. Green here
        # lower_green = np.array([29, 86, 6])
        # upper_green = np.array([64, 255, 255])

        lower_green = 200
        upper_green = 255

        # Gives binary mask  representing where in the image the color “green”
        # mask = cv2.inRange(hsv, lower_green, upper_green)
        #mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue,green,red = cv2.split(frame)

        new_green = np.zeros_like(green)
        canvas = np.zeros_like(green)

        ind_green=np.where(green>240)
        new_green[ind_green] = 255
        #print(new_green.shape)

        
        # Make elliptical kernel
    #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    #     # Opening morph(erosion followed by dilation)to remove small blobs from the mask
    #     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find all contours (i.e. “outlines”) of the objects in the binary mask
        contours, hierarchy = cv2.findContours(new_green.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        # Check to ensure at least one object was found in our frame
        if len(contours) > 0:
            # Find the biggest contour(based on area)
            biggest_contour = max(contours, key=cv2.contourArea)
            if (cv2.contourArea(biggest_contour) < 2000):
                pass
            elif (cv2.contourArea(biggest_contour) > 40000):
                pass
            else:
                moments = cv2.moments(biggest_contour)
                centre_of_contour = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
                cv2.circle(new_green, centre_of_contour, 5, (0, 0, 255), -1)
                # Update the list of center_points  containing the center (x, y)-coordinates of the object so that we draw line tracking it
                center_points.appendleft(centre_of_contour)


            # Find center of contour and draw filled circle
            

    #         # Bound the contour with circle
                ellipse = cv2.fitEllipse(biggest_contour)
                cv2.ellipse(frame, ellipse, (0, 255, 255), 2)

        #Loop over the set of tracked points
        for i in range(1, len(center_points)):
            # b = random.randint(230, 255)
            # g = random.randint(100, 255)
            # r = random.randint(100, 255)
            if math.sqrt(((center_points[i - 1][0] - center_points[i][0]) ** 2) + (
                    (center_points[i - 1][1] - center_points[i][1]) ** 2)) <= 70:
                # Draw line from center points of contour
                # cv2.line(frame, center_points[i - 1], center_points[i], (b, g, r), 4)
                cv2.line(frame, center_points[i - 1], center_points[i], (255,255,0), 4)
                cv2.line(canvas, center_points[i - 1], center_points[i], (255,255,255), 20)


        if(counter%200==0):
            list_center_points = np.asarray(center_points)
            x,y,w,h=cv2.boundingRect(list_center_points)
            print(x,y,w,h)
            print(counter)
            #print(cv2.convexHull(list(center_points)))
            # cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
            checkletter(canvas[y-20:y+h+20,x-20:x+w+20])
            cv2.imshow('compImage', canvas)
            cv2.waitKey(1)
            
            counter=0        

        cv2.imshow('original', frame)
    #     cv2.imshow('mask', mask)
        cv2.imshow('green', new_green)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    # cv2.destroyAllWindows()
    cap.release()


