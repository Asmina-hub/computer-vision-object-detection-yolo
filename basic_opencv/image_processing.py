import cv2

img=cv2.imread('asmina.jpeg')
resizing = cv2.resize(img,(640,480))
grey=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurring= cv2.GaussianBlur(img,(51,51),0)
edges=cv2.Canny(img,50,50)

cv2.imshow('resize',resizing)
cv2.imshow('grey',grey)
cv2.imshow('blurred', blurring)
cv2.imshow('edges',edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

