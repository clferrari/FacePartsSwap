
import cv2
from face_swap import face_swap
import argparse

parser = argparse.ArgumentParser(description='Options')

parser.add_argument('--src_path', dest='src_path', action='store_true', default='./images_test/196.jpg', help='Source image path')
parser.add_argument('--dst_path', dest='dst_path', action='store_true', default='./images_test/185.jpg', help='Target image path')
parser.add_argument('--part', dest='part', action='store_true', default='mouth', help='Part to be swapped')
parser.add_argument('--debug', dest='debug', action='store_true', default=False, help='save debug')
parser.add_argument('--cropImg', dest='cropImg', action='store_true', default=False, help='Crop face')

args = parser.parse_args()

device = "cpu"  # "cuda:0"

img_path = args.src_path
img2_path = args.dst_path

part_to_swap = args.part  # nose, eyes, face, mouth, eyebrows

result_path = './results/'
swapped_img, noClone = face_swap(img_path, img2_path, result_path, part_to_swap, visDebug=args.debug, cropImg=args.cropImg)
cv2.imwrite(result_path + 'swapped.jpg', swapped_img)
cv2.imwrite(result_path + 'swapped_raw.jpg', noClone)
