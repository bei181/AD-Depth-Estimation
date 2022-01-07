import cv2
import numpy as np

def main():
    mask = np.zeros((128, 416*3))
    cv2.imwrite('fake_mask_fseg.png',mask)


if __name__ == '__main__':
    main()