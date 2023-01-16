import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, default=None, help="select one of image : data/image.jpg") # image file name
    parser.add_argument("--source", type=str, default=None, help="image path : data/example") # image path
    parser.add_argument("--webcam", type=int, default=None, help="source of webcam : 0 / 1 / 2") # source number
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_opt()
    print(args.img)
