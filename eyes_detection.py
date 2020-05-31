import cv2 as cv
eye_cascade=cv.CascadeClassifier('haarcascade_eye.xml')
img=cv.imread('eyes.jpg')
img=cv.resize(img,(700,500))
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
eyes=eye_cascade.detectMultiScale(gray,1.9,5)

for x,y,w,h in eyes:
	cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)


cv.imshow('EYES',img)
cv.waitKey(0)
cv.destroyAllWindows()
