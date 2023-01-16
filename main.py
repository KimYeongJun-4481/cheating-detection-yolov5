import os
import cv2
import time
import torch
import argparse
from datetime import datetime
from PIL import Image
from pathlib import Path

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, default=None, help="select one of image : data/image.jpg") # image file name
    parser.add_argument("--source", type=str, default=None, help="image path : data/example") # image path
    parser.add_argument("--webcam", type=int, default=None, help="source of webcam : 0 / 1 / 2") # source number
    parser.add_argument("--weights", type=str, default="s", help="select s / m / l") # yolov5s / yolov5m / yolov5l
    parser.add_argument("--device", type=str, default="cuda", help="select cpu / cuda / mac") # CPU / CUDA / MAC
    parser.add_argument("--save_path", type=str, default=None, help="save results of images : first_test") # save path
    args = parser.parse_args()
    return args

def main():
    args = parse_opt() # arguments
    save = Path("results") # 저장 경로의 상위 경로
    # boundging box에 사용할 색
    colors = {"person"     : [255, 189, 51],
              "tv"         : [41, 251, 213],
              "laptop"     : [243, 85, 48],
              "cell phone" : [121, 56, 244],
              "book"       : [46, 240, 49]}
    classes = list(colors.keys()) # 클래스들의 영어 이름
    class_names = ["사람", "TV", "노트북", "휴대폰", "책"] # 클래스들의 한글 이름
        
    # 만약 세가지 옵션 중 하나라도 선택하지 않으면 예외 발생
    assert not (args.img is None and args.source is None \
        and args.webcam is None), "Few arguments to execute program"

    if args.device == "cuda": # cuda를 선택할 경우
        assert torch.cuda.is_available, "CUDA is not available. set --device cpu instead" # cuda가 사용 불가능하다면
        device = torch.device("cuda:0")
    elif args.device == "mac": 
        device = torch.device("mps") # m1 mac
    else:
        device = torch.device("cpu") # cpu
        
    if args.img is not None or args.source:
        # 저장 경로를 선택하지 않을 경우 현재 시각을 이름으로 가진 폴더를 생성
        if not os.path.isdir(save): os.mkdir(save)
        if args.save_path is None : save_path = save / Path(datetime.now().strftime("%Y%m%d_%H%M%S"))
        else: save_path = args.save_path
        os.mkdir(save_path)
        t = time.time() # 시작 시간 측정
    
    # detection model : yolov5s, yolov5m, yolov5l
    model = torch.hub.load('ultralytics/yolov5', f'yolov5{args.weights}')
    model = model.to(device)
    
    if args.img is not None: # 하나의 이미지
        res = model(Image.open(args.img)) # detection
        # xmin, ymin, xmax, ymax, confidence, classes -> DataFrame의 형태로
        pred = res.pandas().xyxy[0]   
        
        print(pred)  
        
        res = res.render()[0] # render results
        res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB) # 컬러 채널 변환
        cv2.imwrite(save_path / Path(args.img), res) # bounding box가 포함된 결과를 이미지로 저장
        print(f"Time : {(time.time() - t):.2f}s") # 총 소요 시간 측정
                  
    elif args.source is not None: # 폴더(여러개의 이미지)
        pass
    elif args.webcam is not None: # 실시간 웹캠
        pass
    
if __name__ == "__main__":
    main()
