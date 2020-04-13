import cv2

## Fotoğrafı oku ##
img = cv2.imread("TestImages/face.png")

## Cascade dosyasını oku ##
face_cascade = cv2.CascadeClassifier("HaarCascadeFiles/haarcascade_frontalcatface.xml")

## Resmi griye çevir ##
grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

## Yüz tespit işlemi: fotoğrafta bulunan yüzün ya da yüzlerin koordinatlarını verir
#  2.parametre 1.3 oranında küçülttüm ölçeklendirme
#  3.parametre belli bir bölgede 4 farklı pencerede yüz bulsun ki oranın yüz olduğundan emin olalım
#  faces değişkeni içerisinde bir tuple oluşur. 4 adet değişken oluşur, ilk ikisi yüzün sol üst koordinatını
#  diğer ikisi yükseklik ve genişliği temsil eder##
faces = face_cascade.detectMultiScale(grayImg,1.3,6)
print(faces)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

## Fotoğrafı göster ##
cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()