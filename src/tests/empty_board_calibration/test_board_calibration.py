
import pathlib
import time
from typing import List

import cv2 as cv
import matplotlib
import numpy as np

from data_structures.score_segments import ScoreSegment, SegmentType
from fiddling.pipelines.rg_start import get_board_rg_start
from operations_on.contours import get_center_of_contour

matplotlib.rcParams["figure.dpi"] = 600
from matplotlib import pyplot as plt

from calculate_score_board import get_board

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
    try:
        board_rets = get_board_rg_start(empty_board_image)
    except AssertionError as e:
        print(f"Calibration of {image_name} failed.")
        print(e)
        board_rets = (False, [])

    end_time = time.time_ns()
    calibration_completed = board_rets[0]
    if calibration_completed:
        time_needed = round((end_time - start_time) / 1000000, 2)
        print(f"Calibration of {image_name} completed in {time_needed} ms.")
        calibrations_completed += 1
        times.append(time_needed)
        if output_successful_plots:
            score_segments: List[ScoreSegment] = board_rets[1]
            contours = [board_segment.contour for board_segment in score_segments]
            cv.drawContours(empty_board_image, contours, -1, (0, 255, 255), 2)
            for score_segment in score_segments:
                number = score_segment.number
                center = get_center_of_contour(score_segment.contour)
                type_str = score_segment.segment_type.name
                if score_segment.segment_type == SegmentType.OUTER:
                    cv.putText(empty_board_image, str(number), (center.x, center.y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            f, ax = plt.subplots(1, 1)
            plt.title(image_name)
            empty_board_image = cv.cvtColor(empty_board_image, cv.COLOR_BGR2RGB)
            ax.imshow(empty_board_image)
            f.tight_layout()
            plt.show()

print(f"{calibrations_completed}/{all_images_quantity} calibrations completed.")
print(f"Average completion time: {round((sum(times) / len(times)), 2)} ms")
