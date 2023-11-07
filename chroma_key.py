import cv2
import numpy as np

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