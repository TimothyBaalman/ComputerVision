import cv2

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Defining a function that will do the detections
def detectFace(grayScaledImage, orginalFrame):
	# xy are coordinates in upper left corner
	# w is the width of the rectangle
	# h is the height of the rectangle
	reduceScaleFactor = 1.3
	minNumOfNeighborsNeededToBeAccepted = 5
	
	# gives coordinates of x,y,w, and h of image
	faces = face_cascade.detectMultiScale(grayScaledImage, reduceScaleFactor, minNumOfNeighborsNeededToBeAccepted)
	for (x, y, w, h) in faces:
		upperLeftCorner = (x, y)
		bottomRightCorner = (x+w, y+h)
		color = (255, 0, 0)
		thickness = 2
		# also has linetype and shift
		cv2.rectangle(orginalFrame, upperLeftCorner, bottomRightCorner, color, thickness)
		regionOfIntrestForGrayScale = grayScaledImage[y:y+h, x:x+w]
		regionOfIntrestForOrginalColor = orginalFrame[y:y+h, x:x+w]
		eyeReduceScaleBy = 1.1
		eyeMinNumOfNeighbors = 3
		eyes = eye_cascade.detectMultiScale(regionOfIntrestForGrayScale, eyeReduceScaleBy, eyeMinNumOfNeighbors)
		for (eyeX, eyeY, eyeW, eyeH) in eyes:
			cv2.rectangle(regionOfIntrestForOrginalColor, (eyeX, eyeY), (eyeX+eyeW, eyeY+eyeH), (0, 255, 0), 2)

	return orginalFrame # the previous code added the rectangles to the origFame

# Doing Face Recognition with a webcam
video_capture = cv2.VideoCapture(0) # 0 if computers camera 1 if external camera
while True:
	_, lastFrame = video_capture.read() # we only want the second element and we don't want the first
	grayScaledLastFrame = cv2.cvtColor(lastFrame, cv2.COLOR_BGR2GRAY)
	canvas = detectFace(grayScaledLastFrame, lastFrame)
	# display outputs
	cv2.imshow("Video", canvas)
	if cv2.waitKey(1) & 0xFF == ord('q'): # break out of loop when q is pressed
		break
video_capture.release()
cv2.destroyAllWindows()