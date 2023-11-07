## Applying chroma key with OpenCV & Python ##

### Main idea ###

To effectively apply a chroma key, first we need to isolate the values of the color we want to remove.
While RGB spectrum is quite common, I find HSV spectrum more effective to perform this task, so the algorithm
will be based on two blocks (and files):

-First file contains an HSV detector GUI, so we can load a sample of our chroma and, play with the
hue, saturation and value and find out which parameters HSV we need to isolate.

-Second file will apply the chroma key isolation to a given video.

### Import main modules ###

Pretty much all we'll need is OpenCV, Numpy and the Tkinter filedialog.


```python
import cv2
import numpy as np
from tkinter import filedialog
```


    

### HSV Selector ###


```python
class HSV_MODIFIER:

    def __init__(self, path):

        self.file = path

        # Load image
        image = cv2.imread(self.file)
        original_width, original_height = image.shape[:2] # Saving original shape
        image = cv2.resize(image, (600,600)) # Resize to a workable shape if its too big

        # Create a window for the trackbars
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
            # Quit
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            # Save the image, if wanted.
            elif cv2.waitKey(10) & 0xFF == ord('s'):
                new_image_path = filedialog.asksaveasfilename(defaultextension='.jpg', filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
                new_img = cv2.resize(result, (original_height, original_width))
                cv2.imwrite(new_image_path, new_img)
                continue

        cv2.destroyAllWindows()
    # Empty function to fill the trackbars changes
    def nothing(self, x):
        pass

HSV_MODIFIER('./greenfire_sample.png')
```

##Isolating green color (or whatever needed)##

![isolation_green](https://github.com/RomeroRodriguezD/Chroma-Key-with-OpenCV/assets/105886661/345da6aa-a8aa-4538-aedc-e3bfb38303ee)

### Chroma key function ###

```python
def chroma_key(image):

    # Loading image. The resizing is just for demonstration purposes.
    img = image
    img = cv2.resize(img, (600, 600))

    # HSV conversion
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Here we define the green color threshold based on our previous analysis
    lower_green = np.array([30, 200, 120])
    upper_green = np.array([80, 255, 255])

    # Masking the desired space. This lets us to fill the green background with whatever we want to
    mask = cv2.inRange(hsv, lower_green, upper_green)
    masked_image = np.copy(img)
    masked_image[mask != 0] = [0, 0, 0]

    # Background image of our choice
    background_image = cv2.imread('./noucamp.jpg')

    background = cv2.resize(background_image, (600,600))
    background[mask == 0] = [0, 0, 0] # Applying the mask to our background, so it will fill the previously green space
    final_image = background + masked_image # Mixing the new background with our image
    return final_image
```
### Wrapping it up: chroma-keyed video ###

Here we'll mix this fire with green chroma:

<p align="center">
  <img src="https://github.com/RomeroRodriguezD/Chroma-Key-with-OpenCV/assets/105886661/deb2c89f-9faf-4798-bc0c-c416711eaf49" style="width: 50%; height: auto;" />
</p>

With the Camp Nou Stadium:
<p align="center">
  <img src="https://github.com/RomeroRodriguezD/Chroma-Key-with-OpenCV/assets/105886661/8b586abe-f197-4494-b27e-ebf5877e5181" style="width: 50%; height: auto;" />
</p>


```python
if __name__ == "__main__":
    video = './flamegreen.mp4'
  cap = cv2.VideoCapture(video)

  while cap.isOpened():
      ret, frame = cap.read()
      # if frame is read correctly ret is True
      if not ret:
          print("Can't receive frame (stream end?). Exiting ...")
          break
      frame = chroma_key(frame)
      cv2.imshow('frame', frame)
      if cv2.waitKey(1) == ord('q'):
          break
  
  cap.release()
  cv2.destroyAllWindows()
```

## Result ##
<p align="center">
  <img src="https://github.com/RomeroRodriguezD/Chroma-Key-with-OpenCV/assets/105886661/fdb9b8b8-2acc-412e-8123-52b7eae8e973" />
</p>
