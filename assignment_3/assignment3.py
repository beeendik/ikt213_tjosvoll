import cv2
import numpy as np
import matplotlib.pyplot as plt


def sobel_edge_detection(image):

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    blurred = cv2.GaussianBlur(gray, ksize=(3, 3), sigmaX=0)
    sobel = cv2.Sobel(blurred, cv2.CV_64F, dx=1, dy=1, ksize=1)
    sobel_edges = cv2.convertScaleAbs(sobel)

    return sobel_edges


def canny_edge_detection(image, threshold_1=50, threshold_2=50):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    blurred = cv2.GaussianBlur(gray, ksize=(3, 3), sigmaX=0)
    edges = cv2.Canny(blurred, threshold_1, threshold_2)

    return edges

def template_match(image, template):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if len(template.shape) == 3:
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    else:
        gray_template = template.copy()

    w, h = gray_template.shape[::-1]
    res = cv2.matchTemplate(gray, gray_template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where(res >= threshold)
    result_img = image.copy()

    for pt in zip(*loc[::-1]):
        cv2.rectangle(result_img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    return result_img

def resize(image, scale_factor: int, up_or_down:str):
    resized_img = image.copy()

    if up_or_down.lower() == 'ask':
        cv2.imshow("Press i to zoom in, o to zoom out", image)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

    if key == ord('i'):
        up_or_down = "up"
    elif key == ord('o'):
        up_or_down = "down"
    else:
        print("Not a valid key")
        return image

    if up_or_down.lower() == "up":
        for _ in range(scale_factor):
            resized_img = cv2.pyrUp(resized_img)
    elif up_or_down.lower() == "down":
        for _ in range(scale_factor):
            resized_img = cv2.pyrDown(resized_img)
    else:
        raise ValueError("")

    save_path = f"resized_{up_or_down}_{scale_factor}.png"
    cv2.imwrite(save_path, resized_img)
    print(f"Resized image saved at: {save_path}")

    return resized_img

def main():

    img_path = "lambo.png"
    img = cv2.imread(img_path)

    if img is None:
        print(f"Error: Could not read image from {img_path}")
        return

    gray_path = "shapes-1.png"
    gray = cv2.imread(gray_path)
    template_path = "shapes_template.jpg"
    template = cv2.imread(template_path)
    if template is None:
        print(f"Error: could not read template fro {template_path}")
        return

    #Sobel
    sobel_edges = sobel_edge_detection(img)
    sobel_save_path = "sobel_edges.png"
    cv2.imwrite(sobel_save_path, sobel_edges)
    print(f"Sobel edge-detected image saved at: {sobel_save_path}")

    #canny
    canny_edges = canny_edge_detection(img, threshold_1=50, threshold_2=50)
    canny_save_path = "canny_edges.png"
    cv2.imwrite(canny_save_path, canny_edges)
    print(f"Canny edge-detected image saved at: {canny_save_path}")

    #MATCHED IMAGE TEMPLATE
    matched_img = template_match(gray, template)
    template_path = "template.png"
    cv2.imwrite(template_path, matched_img)
    print(f"Template matched image saved at: {template_path}")

    #RESIZED
    resize(img, scale_factor=2, up_or_down="ask")


if __name__ == "__main__":
    main()
