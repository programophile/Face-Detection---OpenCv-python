import cv2
face_cap=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video= cv2.VideoCapture(0)
while True:
  _,video_frame=video.read()
  changed_color=cv2.cvtColor(video_frame,cv2.COLOR_BGR2GRAY)
  face_detect=face_cap.detectMultiScale(changed_color,1.1,5)
  for (x,y,w,h) in face_detect:
    cv2.rectangle(video_frame,(x,y),(x+w,y+h),(255, 0, 255), 2)
  cv2.imshow("face detection by sad",video_frame)
  if cv2.waitKey(10)==27:
    break

video.release()