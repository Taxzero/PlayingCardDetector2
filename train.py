import torch
from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # CUDA가 사용 가능한 경우 GPU를 사용하도록 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(torch.cuda.is_available())

    # CUDA가 사용 가능한 경우 CUDA 장치 설정
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # 원하는 CUDA 장치 번호 설정

    # YOLO 모델 로드
    model = YOLO('yolov8s.pt').to(device)

    # 데이터셋 경로 설정
    data_path = 'C:/Users/bjkim/repos/PlayingCardDetector2/dataset.yaml'

    # 모델 학습
    model.train(data=data_path, epochs=263, imgsz=640, batch=64, device=device, workers=8)