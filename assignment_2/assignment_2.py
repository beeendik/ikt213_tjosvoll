import cv2
import numpy as np

def padding(image, border_width = 100):
    img = cv2.imread(image)

    reflect = cv2.copyMakeBorder(img, border_width, border_width, border_width, border_width, cv2.BORDER_REFLECT)

    cv2.imwrite('lena_reflect.png', reflect)


def crop(image, x_0, x_1, y_0, y_1):
    img = cv2.imread(image)

    x_start = x_0 + 80
    y_start = y_0 + 80
    x_end = x_1 - 130
    y_end = y_1 - 130

    cropped = img[x_start:x_end, y_start:y_end]

    cv2.imwrite('cropped.png', cropped)


def resize(image, width, height):
    img = cv2.imread(image)

    resized = cv2.resize(img, (width, height))

    cv2.imwrite('resized.png', resized)


def copy(image, emptyPictureArray):
    img = cv2.imread(image)

    height, width, channels = img.shape

    if emptyPictureArray is None or emptyPictureArray.shape != (height, width, 3):
        emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)

    emptyPictureArray[:, :, :] = img[:, :, :]

    cv2.imwrite('lena-copy.png', emptyPictureArray)

    return emptyPictureArray


def grayscale(image):
    img = cv2.imread(image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imwrite('grayscale.png', gray)


def hsv(image):
    img = cv2.imread(image)

    hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    cv2.imwrite('hsv.png', hsvImage)


def hue_shifted(image, emptyPictureArray, hue):
    img = cv2.imread(image)

    height, width, channels = img.shape

    if emptyPictureArray is None or emptyPictureArray.shape != (height, width, 3):
        emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)

    # CV2.ADD ENSURES VALUES UNDER 0 BECOMES 0, AND VALUES OVER
    # 255 BECOMES 255. IF WE WANTED TO ACCEPT VALUES OVER 255 OR UNDER 0
    # WE COULD'VE USED NUMPY MATH (OVERFLOW, UNDERFLOW)
    shift_color = np.full(img.shape, hue, dtype=np.uint8)
    emptyPictureArray[:, :, :] = cv2.add(img, shift_color)

    cv2.imwrite('color-shift.png', emptyPictureArray)

    return emptyPictureArray


def smoothing(image):
    img = cv2.imread(image)

    blur = cv2.GaussianBlur(img, (15, 15), cv2.BORDER_DEFAULT)

    cv2.imwrite('smoothing.png', blur)


def rotation(image, rotation_angle):
    img = cv2.imread(image)

    if rotation_angle == 180:
        rotatedImage = cv2.rotate(img, cv2.ROTATE_180)

    elif rotation_angle == 90:
        rotatedImage = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    else:
        ValueError("rotation angle must be 90 or 180")

    cv2.imwrite(f'rotated-image{rotation_angle}.png', rotatedImage)

    return rotatedImage

def main():
    padding("lena-1.png")
    crop("lena-1.png", 0, 512, 0, 512)
    resize("lena-1.png", 200, 200)
    copy("lena-1.png", None)
    grayscale("lena-1.png")
    hsv("lena-1.png")
    hue_shifted("lena-1.png", None, 50)
    smoothing("lena-1.png")
    rotation("lena-1.png", 90)
    rotation("lena-1.png", 180)


if __name__ == "__main__":
    main()