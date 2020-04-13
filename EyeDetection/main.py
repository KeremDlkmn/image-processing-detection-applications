import cv2

img = cv2.imread("TestImages/face.png")

face_cascade = cv2.CascadeClassifier("HaarCascadeFiles/haarcascade_frontalcatface.xml")
eye_cascade = cv2.CascadeClassifier("HaarCascadeFiles/eye.xml")

grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(grayImg,1.3,6)

faceImg = []
faceGrayImg = []

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)

    ## seçtiğimiz yüzü değişkene atalım, yüzü yakalayalım
    #  x'ten x+w ve y'den y+h'a kadar olan kısımları seçtik y##
    faceImg = img[y:y+h,x:x+w]
    faceGrayImg = grayImg[y:y+h,x:x+w]

eyes = eye_cascade.detectMultiScale(faceGrayImg)

for (ex,ey,ew,eh) in eyes:
    cv2.rectangle(faceImg,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)

cv2.imshow("eyes",img)
cv2.waitKey(0)
cv2.destroyAllWindows()