# torch allows for fast and effiecient computation of the gradient
import torch
# Used to Convert Tensors to torch variables which contains both the tensors and gradient
from torch.autograd import Variable
# Used to draw rectangles on image not using opencv to do detection
import cv2
# Use BaseTransform for formatting Images to be compatible
# Use VOC_CLASSES as a dictionary for encoding object values
from data import BaseTransform, VOC_CLASSES as labelmap
# Used for constuctor for the ssd neural network
from ssd import build_ssd
# Used for processing the images and applying the detect functions (PIL can be used instead, but is longer to type out code)
import imageio

# Defining detection funciton
def detectObjects(imageForDetection, ssdNeuralNetwork, transformImageFormat):
	height, width = imageForDetection.shape[:2] # also returns the color channels 2 for BW and 3 for RGB which is in the third position
	# Transformations
	transformImageToNumpyArray = transformImageFormat(imageForDetection)[0]
	# convert numpy array to torch tensor
	torchTensor = torch.from_numpy(transformImageToNumpyArray).permute(2, 0, 1) #rbg to grb green=2, red=0, blue=1
	# add fake batch dimension the unsqueeze and convert torchTensor batches to torch variable
	torchVar = torchTensor.unsqueeze(0)
	with torch.no_grad():
		# feed torchVars to neural network
		neuralNetOutput = ssdNeuralNetwork(torchVar)

	# contains 4 elements [batch, numberOfClasses or the objects that were detected,
	#  numberOfOccurencesOfTheClass, tupleOfElements (score, x0, y0, x1, y1)] scorce is the like a threshold for an object detected
	detectionsTensor = neuralNetOutput.data

	# first width, height correspond to the scalar values of the  upperleft corner of rectangle detector 
	# the second correspond to the scalor values of the lowerRight corner of rectangle detector
	scale = torch.Tensor([width, height, width, height]) # This is done to normalize the scalar values of the detected object
	# between 0 and 1.

	for aClass in range(detectionsTensor.size(1)):
		occurance = 0
		while detectionsTensor[0, aClass, occurance, 0] >= 0.3:
			# gets coordinate at scale
			point = (detectionsTensor[0, aClass, occurance, 1:] * scale).numpy() # uses every tupleElement using 1: and convert back to numpy to work with opencv
			upperLeftCorner = int(point[0]), int(point[1])
			lowerRightCorner = int(point[2]), int(point[3])
			color = (255, 0, 0)
			thickness = 2
			# draw rectangle on image
			cv2.rectangle(imageForDetection, upperLeftCorner, lowerRightCorner, color, thickness)
			labelToPrint =  labelmap[aClass - 1]
			positionToDisplayText = upperLeftCorner
			font = cv2.FONT_HERSHEY_SIMPLEX
			textSize = 2
			textColor = (255, 255, 255)
			textThickness = 2
			# print label on the rectangle
			cv2.putText(imageForDetection, labelToPrint, positionToDisplayText, font, textSize, textColor, textThickness, cv2.LINE_AA)
			occurance += 1
	return imageForDetection

# Create SSD Neural Network
neuralNetwork = build_ssd('test')
tensorWithWeights = torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage)
neuralNetwork.load_state_dict(tensorWithWeights)

# Create Transformation
colorScaleTransform = (104/256.0, 117/256.0, 123/256.0)
transform = BaseTransform(neuralNetwork.size, colorScaleTransform)

# Detect Objects in video frames
reader = imageio.get_reader('epic_horses.mp4')
frequenceFPS = reader.get_meta_data()['fps']
writer = imageio.get_writer('epic_horses_with_detection.mp4', fps = frequenceFPS)
for frameIndex, frame in enumerate(reader):
	frame = detectObjects(frame, neuralNetwork.eval(), transform)
	writer.append_data(frame)
	print(frameIndex)
writer.close()