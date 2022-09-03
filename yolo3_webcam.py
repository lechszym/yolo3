import os
from imageai.Detection import ObjectDetection
import cv2
import urllib.request

# Load object detection model from IMAGEAI library
obj_detect = ObjectDetection()
# Set the model type to YOLO3
obj_detect.setModelTypeAsYOLOv3()

trainedYoloBinary = './yolo.h5'
trainedYoloURL = 'https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5'
#Check for the binary of the trained weights
if not os.path.exists(trainedYoloBinary):
   #If it's not there, download it:
   print("Downloading YOLO model binaries...")
   response = urllib.request.urlretrieve(trainedYoloURL, trainedYoloBinary)

# Load the model weights
obj_detect.setModelPath(trainedYoloBinary)
obj_detect.loadModel()


# Size of the image from webcam
# (the bigger, the longer it takes for the model to process images
frame_width = 416
frame_height = 416

# Open the webcam feed
cam_feed = cv2.VideoCapture(0)
cam_feed.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cam_feed.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

while True:
   # Read the next image from the webcam
   ret, img = cam_feed.read()

   # Detect objects in the images
   annotated_image, preds = obj_detect.detectObjectsFromImage(input_image=img,
                                                              input_type="array",
                                                              output_type="array",
                                                              display_percentage_probability=False,
                                                              display_object_name=True)

   # Show the image with annotation
   cv2.imshow("", annotated_image)

   # Quit if q is presed
   if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1) == 27):
      break

# Clean up
cam_feed.release()
cv2.destroyAllWindows()