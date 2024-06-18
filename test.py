import cv2
import numpy as np
from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO('C:/Users/syoun/repos/PlayingCardDetector2/runs/detect/train5/weights/best.pt')  # 학습된 모델 경로

def four_point_transform(image, pts):
    rect = np.array(pts, dtype="float32")
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

# 웹캠에서 실시간 프레임 읽기
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # YOLOv8 모델을 사용하여 카드 감지
    results = model(frame)
    detections = results[0].boxes  # 감지된 경계 상자
    
    for detection in detections:
        xmin, ymin, xmax, ymax = map(int, detection.xyxy[0])
        confidence = detection.conf[0]
        class_id = detection.cls[0]
        
        # 경계 상자 그리기
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        # 라벨 텍스트 추가
        label = f'{model.names[int(class_id)]}: {confidence:.2f}'
        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        pts = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype="float32")
        
        # 카드 정렬
        aligned_card = four_point_transform(frame, pts)
        
        # 정렬된 카드 이미지를 화면에 표시
        cv2.imshow("Aligned Card", aligned_card)
    
    # 원본 프레임 화면에 표시
    cv2.imshow("Frame", frame)
    
    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
