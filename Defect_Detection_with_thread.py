from threading import Thread
import cv2, time
import numpy as np
import imutils
GOOD_MATCH_PERCENT = 0.7

class VideoStreamWidget(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.Master_image = cv2.imread("Master_image.jpg")
        
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.Test_frame) = self.capture.read()
                
                self.Test_frame = cv2.bilateralFilter(self.Test_frame,9,75,75)
                self.Master_image = cv2.bilateralFilter(self.Master_image,9,75,75)
                
                self.im1Gray = cv2.cvtColor(self.Test_frame, cv2.COLOR_BGR2GRAY)
                self.im2Gray = cv2.cvtColor(self.Master_image,cv2.COLOR_BGR2GRAY)
                
                orb = cv2.ORB_create()
                self.keypoints1, self.descriptors1 = orb.detectAndCompute(self.im1Gray, None)
                self.keypoints2 , self.descriptors2 = orb.detectAndCompute(self.im2Gray,None)
                
                matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
               
                self.matches = matcher.match(self.descriptors1, self.descriptors2, None)
                
                # Sort matches by score
                self.matches.sort(key=lambda x: x.distance, reverse=False)

                # Remove not so good matches
                self.numGoodMatches = int(len(self.matches) * GOOD_MATCH_PERCENT)
                self.matches = self.matches[:self.numGoodMatches]
                # Extract location of good matches
                self.points1 = np.zeros((len(self.matches), 2), dtype=np.float32)
                self.points2 = np.zeros((len(self.matches), 2), dtype=np.float32)

                for self.i, self.match in enumerate(self.matches):
                    self.points1[self.i, :] = self.keypoints1[self.match.queryIdx].pt
                    self.points2[self.i, :] = self.keypoints2[self.match.trainIdx].pt

                # Find homography
                self.h, self.mask = cv2.findHomography(self.points1, self.points2, cv2.RANSAC)

                # Use homography
                self.height, self.width, self.channels = self.Test_frame.shape
                self.im1Reg = cv2.warpPerspective(self.Test_frame, self.h, (self.width, self.height))
                self.diff = cv2.absdiff(self.im1Reg, self.Test_frame)

                self.threshold = 25
                self.im1Reg[np.where(self.diff >  self.threshold)] = 255
                self.im1Reg[np.where(self.diff <= self.threshold)] = 0
                
                
                self.img_bw = 255*(cv2.cvtColor(self.im1Reg, cv2.COLOR_BGR2GRAY) > 15).astype('uint8')
                self.se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
                self.se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    
    
                self.mask = cv2.morphologyEx(self.img_bw, cv2.MORPH_CLOSE, self.se1)
                self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, self.se2)
                self.thresh = cv2.threshold(self.mask, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                self.cnts = cv2.findContours(self.thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                self.cnts = imutils.grab_contours(self.cnts)
                for self.c in self.cnts:
                    (self.x,self.y, self.w, self.h) = cv2.boundingRect(self.c)
                    self.area= cv2.contourArea(self.c)
                    cv2.rectangle(self.Test_frame, (self.x, self.y), (self.x +self.w, self.y + self.h), (0, 0, 255), 2)
    

            time.sleep(.01)

    def show_frame(self):
        # Display frames in main program
        cv2.imshow('frame', self.Test_frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)
    
    
if __name__ == '__main__':
    video_stream_widget = VideoStreamWidget()
    while True:
        try:
            video_stream_widget.show_frame()
        except AttributeError:
            pass