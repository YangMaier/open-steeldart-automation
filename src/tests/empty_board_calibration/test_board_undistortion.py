import pathlib
import time

import cv2 as cv

from calibrate_cam_with_dartboard import calibrate_cam_with_dartboard

# empty_boards_folder_path = pathlib.Path(__file__).parent.joinpath("empty_boards")
empty_boards_folder_path = pathlib.Path(__file__).parent.joinpath("empty_boards")

calibrations_completed = 0
all_images_quantity = 0
times = []

output_successful_plots = True
output_failed_plots = False

for image_path in empty_boards_folder_path.iterdir():
    image_name = image_path.name
    all_images_quantity += 1
    empty_board_image = cv.imread(str(image_path))
    start_time = time.time_ns()
    # board_rets = get_board(empty_board_image)
    rets = calibrate_cam_with_dartboard(empty_board_image)
