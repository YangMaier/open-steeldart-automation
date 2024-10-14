#   capture = cv2.VideoCapture(self.cam_id, cv2.CAP_V4L2)
#   capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
#   width = 1920
#   height = 1080
#   capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#   capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
def make_1080p(cap):
    cap.set(3, 1920)
    cap.set(4, 1080)


def make_720p(cap):
    cap.set(3, 1280)
    cap.set(4, 720)


def make_480p(cap):
    cap.set(3, 640)
    cap.set(4, 480)
