import cv2
import dlib

desired_width = 469  # желаемая ширина
desired_height = 640  # желаемая высота
dim = (desired_width, desired_height)  # размер в итоге

x = []
y = []
i = 0
points = [7, 20, 30, 36, 39, 42, 45] # 7 - край брови; 20 - край челюсти ;30 - кончик носа;
                                     # 36, 39 - углы левого глаза; 42, 45 - углы правого глаза

# Load the detector
detector = dlib.get_frontal_face_detector()
# Load the predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# read the image
img = cv2.imread("ronaldu-13.jpg")
# Convert image into grayscale
gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
# Use detector to find landmarks
faces = detector(gray)
for face in faces:
    x1 = face.left() # left point
    y1 = face.top() # top point
    x2 = face.right() # right point
    y2 = face.bottom() # bottom point
    # Look for the landmarks
    landmarks = predictor(image=gray, box=face)

    for n in points:
        x.append(landmarks.part(n).x)
        y.append(landmarks.part(n).y)
        # Draw a circle

        i+=1

xres=x[1]+x[2]
xres = xres//2
cv2.line(img, ((x[3]+x[4])//2, y[0]), ((x[3]+x[4])//2, y[1]), (0, 255, 0), 2) #левая побочная линия симметрии
cv2.line(img, ((x[5]+x[6])//2, y[0]), ((x[5]+x[6])//2, y[1]), (0, 255, 0), 2) #правая побочная линия симметрии
cv2.line(img, (x[2], y[0]), (x[2], y[1]//2), (0, 255, 0), 2) #основная линия симметрии
cv2.putText(img, 'r = ' + str(((x[5]+x[6])//2)-x[2]), (x[3], y[1]), cv2.FONT_HERSHEY_SIMPLEX,
             0.75, (0, 255, 0), 2)
cv2.putText(img, 'r = ' + str(x[2]-((x[3]+x[4])//2)), (x[5], y[1]), cv2.FONT_HERSHEY_SIMPLEX,
             0.75, (0, 255, 0), 2)
#img = cv2.resize(img, dim)
# show the image
cv2.imshow(winname="Face", mat=img)
# Wait for a key press to exit
cv2.waitKey(delay=0)
# Close all windows
cv2.destroyAllWindows()