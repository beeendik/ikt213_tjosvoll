import cv2

def print_image_information(image):

    img = cv2.imread(image)

    height, width, channels = img.shape
    size = img.size
    datatype = img.dtype

    print("height: ", height)
    print("width: ", width)
    print("channels: ", channels)
    print("size: ", size, img.shape)
    print("data type: ", datatype)


print_image_information("lena-1.png")

#####------#####

def print_video_information():

    #åpner standardkamera
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # tester en frame for å se om kamera funker
    ret, frame = cam.read()
    if not ret:
        print("couldn't open camera")
        return

    #henter kamerainfo
    fps = cam.get(cv2.CAP_PROP_FPS)
    height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)

    #skirver kamerainfo til txt fil
    with open("solutions/camera_outputs.txt", "w") as f:
        f.write(f"fps: {fps}\n")
        f.write(f"height: {height}\n")
        f.write(f"width: {width}\n")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to get frame")
            break

        #viser frame i et vindu til bruker
        cv2.imshow("camera", frame)

        #avslutt på "q"
        if cv2.waitKey(1) == ord('q'):
            break


    cam.release()
    cv2.destroyAllWindows()

print_video_information()





