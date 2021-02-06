# Importing the Libraries
import cv2

# capturing frame from a video
video = cv2.VideoCapture('videoplayback.mp4')

# Pre-trained car classifier
classifier = cv2.CascadeClassifier('cars.xml')

# To capture all frames in the video
while True:
    # read frames from the video
    ret, frames = video.read()
    # convert images to black and white
    gray = cv2.cvtColor(frames,cv2.COLOR_BGR2GRAY)
    # to detect different cars from the capture image
    cars = classifier.detectMultiScale(gray,1.1,5)
    # to draw the rectangles
    for (x,y,w,h) in cars:
        cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2)
    # Display the frames in a window
    cv2.imshow('Car Detection',frames)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()


