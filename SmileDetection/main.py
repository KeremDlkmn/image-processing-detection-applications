import cv2

img = cv2.imread("TestImages/smile.jpg")

face_cascade  = cv2.CascadeClassifier("HaarCascadeFiles/haarcascade_frontalcatface.xml")
smile_cascade = cv2.CascadeClassifier("HaarCascadeFiles/smile.xml")

grayImage = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(grayImage,1.3,6)

faceImg     = []
faceGrayImg = []

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)

    ## Algılanan yüz kısmını yeni değişkene atadım ##
    faceImg     = img[y:y+h,x:x+w]
    faceGrayImg = grayImage[y:y+h,x:x+w]

smiles = smile_cascade.detectMultiScale(faceGrayImg,1.5,7)

for (sx,sy,sw,sh) in smiles:
    cv2.rectangle(faceImg,(sx,sy),(sx+sw,sy+sh),(255,0,0),2)

cv2.imshow("smile",img)
cv2.waitKey(0)
cv2.destroyAllWindows()