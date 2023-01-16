import torch
import argparse
from pathlib import Path

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, default=None, help="select one of image : data/image.jpg") # image file name
    parser.add_argument("--source", type=str, default=None, help="image path : data/example") # image path
    parser.add_argument("--webcam", type=int, default=None, help="source of webcam : 0 / 1 / 2") # source number
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

    if args.device == "cuda": # cuda를 사용할 경우
        assert torch.cuda.is_available, "CUDA is not available" # cuda가 사용 불가능하다면
        device = torch.device("cuda:0")
    elif args.device == "mac": 
        device = torch.device("mps") # m1 mac
    else:
        device = torch.device("cpu") # cpu
    
    if args.img is not None: # 하나의 이미지
        pass                
    elif args.source is not None: # 폴더(여러개의 이미지)
        pass
    elif args.webcam is not None: # 실시간 웹캠
        pass
    
if __name__ == "__main__":
    main()
