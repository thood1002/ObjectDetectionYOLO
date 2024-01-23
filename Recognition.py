import cv2

#Facial recognition cascade import
model=cv2.CascadeClassifier(r"C:\Users\Taylor Hood\Desktop\FacialRecognition\haarcascade_frontalface_default.xml")

#Initializing the webcam we want to capture
webcam=cv2.VideoCapture(0)

#While loop to read frames from our webcam
while True:
    ret,img=webcam.read()

    #Detecting faces
    faces=model.detectMultiScale(img)

    #Creating the yellow boxes for faces 
    for x, y, width, height in faces:
        img=cv2.rectangle(img,(x,y),(x+width,y+height),(0,255,255),2)

    #Displaying image with boxed faces titled "Boxed"
    cv2.imshow("Boxed",img)

    #Waiting before closing boxed image until q is pressed
    key = cv2.waitKey(1)
    if key==ord("q"):
        break
#Releasing webcam after we are done with it
webcam.release()
#Closing boxed image
cv2.destroyAllWindows()