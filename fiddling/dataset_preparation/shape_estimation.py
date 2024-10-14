import pathlib

# use the time series data to train a net that can estimate the shape of a partial visible dart

# dart is partially visible on cam 1
# dart is fully visible on cam 2

# get DartBoard for both cams
# match template on cam 2
# warp abstract template data from cam 2 to cam 1

# get shape bounding box for cam 1

# input is bounding box img GEHT SO NICHT WEIL UNVOLLSTÄNDIG
# output is abstract template data

img_empty_board_path = pathlib.Path().absolute().joinpath("../media/fiddling/empty_board_1080p_distance_mid.png")
dart_board = calculate_score_board(frame)