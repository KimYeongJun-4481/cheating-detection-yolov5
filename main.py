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
    
    print(classes)
    
if __name__ == "__main__":
    main()
