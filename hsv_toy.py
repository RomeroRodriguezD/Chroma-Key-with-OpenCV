import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

class HSV_MODIFIER:

    def __init__(self, path):

        self.file = path

        # Load image
        image = cv2.imread(self.file)
        original_width, original_height = image.shape[:2] # Mantener los valores originales para volver a re-shapear antes de dar salida
        image = cv2.resize(image, (600,600)) # Aplicar resize si es muy pequeña o muy grande, para que la GUI no se vaya al carajo
        #image = cv2.resize(image, (540,500)) # Aplicar resize si es muy pequeña o muy grande, para que la GUI no se vaya al carajo

        # Create a window
        cv2.namedWindow('image')
        cv2.resizeWindow('image', 900, 500)
        # Create trackbars for color change
        # Hue is from 0-179 for Opencv
        cv2.createTrackbar('HMin', 'image', 0, 179, self.nothing)
        cv2.createTrackbar('SMin', 'image', 0, 255, self.nothing)
        cv2.createTrackbar('VMin', 'image', 0, 255, self.nothing)
        cv2.createTrackbar('HMax', 'image', 0, 179, self.nothing)
        cv2.createTrackbar('SMax', 'image', 0, 255, self.nothing)
        cv2.createTrackbar('VMax', 'image', 0, 255, self.nothing)

        # Set default value for Max HSV trackbars
        cv2.setTrackbarPos('HMax', 'image', 179)
        cv2.setTrackbarPos('SMax', 'image', 255)
        cv2.setTrackbarPos('VMax', 'image', 255)

        # Initialize HSV min/max values
        hMin = sMin = vMin = hMax = sMax = vMax = 0
        phMin = psMin = pvMin = phMax = psMax = pvMax = 0

        while(1):
            # Get current positions of all trackbars
            hMin = cv2.getTrackbarPos('HMin', 'image')
            sMin = cv2.getTrackbarPos('SMin', 'image')
            vMin = cv2.getTrackbarPos('VMin', 'image')
            hMax = cv2.getTrackbarPos('HMax', 'image')
            sMax = cv2.getTrackbarPos('SMax', 'image')
            vMax = cv2.getTrackbarPos('VMax', 'image')

            # Set minimum and maximum HSV values to display
            lower = np.array([hMin, sMin, vMin])
            upper = np.array([hMax, sMax, vMax])

            # Convert to HSV format and color threshold
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            result = cv2.bitwise_and(image, image, mask=mask)

            # Print if there is a change in HSV value
            if((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
                print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
                phMin = hMin
                psMin = sMin
                pvMin = vMin
                phMax = hMax
                psMax = sMax
                pvMax = vMax

            # Display result image
            cv2.imshow('result', result)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            elif cv2.waitKey(10) & 0xFF == ord('s'):
                new_image_path = filedialog.asksaveasfilename(defaultextension='.jpg', filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
                new_img = cv2.resize(result, (original_height, original_width))
                cv2.imwrite(new_image_path, new_img)
                continue

        cv2.destroyAllWindows()

    def nothing(self, x):
        pass

HSV_MODIFIER('./flamegreen.bmp')