import os
from imageai.Detection import ObjectDetection
import cv2
import urllib.request

images = ['IMG_0014.JPG','IMG_0015.JPG']
im_width_for_yolo = 720
im_height_for_yolo = 480


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

for imgFile in images:

   annotated_image, preds = obj_detect.detectObjectsFromImage(input_image=imgFile,
                                                           input_type="file",
                                                           output_type="array",
                                                           display_percentage_probability=False,
                                                           display_object_name=True)

   # Show the image with annotation
   cv2.imshow("", annotated_image)

   # Wait for any key pressed
   cv2.waitKey(0)
   cv2.destroyAllWindows()