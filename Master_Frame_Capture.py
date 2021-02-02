import cv2


cap = cv2.VideoCapture(0)
i=1
while (i<2):
    _,Test = cap.read()
    cv2.imwrite("Master_Image.jpg", Test)
    i=i+1
cap.release()
cv2.destroyAllWindows()