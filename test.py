import cv2
import numpy as np
from keras.models import load_model

# Load mô hình đã được đào tạo (file .keras)
model = load_model('my_model1.keras')

# Hàm dự đoán khuôn mặt trong thời gian thực từ camera
def detect_face_realtime():
    # Mở camera
    cap = cv2.VideoCapture(0)
    name = ["phuong","other"]
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        # Chuyển đổi sang ảnh xám
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Phát hiện khuôn mặt bằng OpenCV
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            # Cắt và resize khuôn mặt thành 50x50x1
            face = gray_frame[y:y+h, x:x+w]
            face = cv2.resize(face, (100, 100))
            
            # Chuẩn hóa giá trị pixel
            face = face / 255.0

            # Dự đoán sử dụng mô hình đã được đào tạo
            predict =model.predict(np.expand_dims(face,axis=0))
            index = np.argmax(predict)

            # Xuất kết quả
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name[index], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Hiển thị hình ảnh với các khuôn mặt đã được phát hiện
        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng tài nguyên camera và đóng cửa sổ
    cap.release()
    cv2.destroyAllWindows()

# Gọi hàm để bắt đầu nhận diện khuôn mặt trong thời gian thực
detect_face_realtime()
