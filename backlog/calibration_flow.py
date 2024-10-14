from enum import Enum, auto

from src.data_structures.ellipse import Ellipse


class CalibrationState(Enum):
    NO_BOARD_ELLIPSE_FOUND = auto()
    ACCUMULATING_ELLIPSE = auto()
    ELLIPSE_ACCUMULATED = auto()
    ACCUMULATING_CPS = auto()
    BOARD_CALIBRATED = auto()


class CalibrationFlow:
    """
    This flows purpose is to get the dartboard segments in a live scenario
    - Currently untested and not used
    - Not part of my bachelor thesis
    - A start of an implementation of my dart calibration and scoring flow diagram

    - Uses the fact that the camera and the dartboard are not likely to move
    - Calibrates once and then only checks if the picture moved or not
    - If a dart impacts or a hand enters the picture, parts of the picture stay the same, so that's not an issue
    - Has Calibration states that can be read
    - An UI could output those states for the user
    - The UI probably should output if the current frames are calibrated or not and make the user act accordingly
    - A Scoring Flow can use the calibration states to know if a score is valid or not
    - Ellipse accumulating gives a better robustness against noise that's just in one frame
    """

    def __init__(self):
        self._current_frame = None
        self._current_frame_ellipse = None
        self._calibration_state = CalibrationState.NO_BOARD_ELLIPSE_FOUND
        self._accumulated_ellipses: [Ellipse] = []
        self._mean_ellipse: Ellipse = Ellipse(0, 0, 0, 0, 0)
        self._accumulated_fields = []
        self._calibrated_board = None

    # public section

    def get_calibration_state(self):
        return self._calibration_state

    def get_calibrated_board(self):
        return self._calibrated_board

    def calibrate_with_frame(self, frame):
        self._current_frame = frame
        self._current_frame_ellipse = None
        ellipse_found = self._ellipse_found_in_frame()
        if ellipse_found:
            self._ellipse_found_flow()
        else:
            self._ellipse_not_found_flow()

    # private section

    def _ellipse_found_in_frame(self):
        ellipse = find_dartboard_outer_ellipse(self._current_frame, detection_settings.double_segment_contour_min_size,
                                               detection_settings.double_segment_contour_max_size,
                                               detection_settings.ellipse_min_size, detection_settings.ellipse_max_size)
        if ellipse is None:
            return False
        else:
            self._current_frame_ellipse = ellipse
            return True

    def _accumulate_ellipse(self):
        # returns True if enough ellipses were accumulated
        self._accumulated_ellipses.append(self._current_frame_ellipse)
        if self._enough_ellipses_accumulated():
            self._calculate_mean_ellipse()

        return self._enough_ellipses_accumulated()

    def _enough_ellipses_accumulated(self):
        return len(self._accumulated_ellipses) == 15

    def _calculate_mean_ellipse(self):
        self._mean_ellipse = get_mean_ellipse(self._accumulated_ellipses)


    def _accumulate_board_center_points(self):
        # returns if enough board center points have been accumulated
        return False

    def _calculate_board_points(self):
        # self._accumulated_fields
        # self._calibrated_board =
        pass

    def _compare_shape_cp_flow(self):
        enough_cps_the_same = self._compare_shape_cps()
        if enough_cps_the_same:
            self._calibration_state = CalibrationState.BOARD_CALIBRATED
        else:
            self._delete_accumulated_cps_and_ellipse()

    def _delete_accumulated_cps_and_ellipse(self):
        self._accumulated_ellipses.clear()
        self._accumulated_fields.clear()
        self._calibration_state = CalibrationState.NO_BOARD_ELLIPSE_FOUND

    def _ellipse_found_flow(self):
        if self._enough_ellipses_accumulated():
            self._ellipse_accumulated_flow()
        else:
            self._ellipse_not_accumulated_flow()

    def _ellipse_not_found_flow(self):
        if self._calibration_state is CalibrationState.BOARD_CALIBRATED:
            self._compare_shape_cp_flow()
        else:
            self._delete_accumulated_cps_and_ellipse()
            self._calibration_state = CalibrationState.NO_BOARD_ELLIPSE_FOUND

    def _ellipse_accumulated_flow(self):
        if self._calibration_state.BOARD_CALIBRATED:
            self._compare_shape_cp_flow()
        else:
            self._board_not_calibrated_flow()
        pass

    def _ellipse_not_accumulated_flow(self):
        enough_ellipses_accumulated = self._accumulate_ellipse()
        if enough_ellipses_accumulated:
            self._calibration_state = CalibrationState.ELLIPSE_ACCUMULATED
        else:
            self._calibration_state = CalibrationState.ACCUMULATING_ELLIPSE

    def _board_not_calibrated_flow(self):
        enough_board_center_points_accumulated = self._accumulate_board_center_points()
        if enough_board_center_points_accumulated:
            self._calculate_board_points()
            self._calibration_state = CalibrationState.BOARD_CALIBRATED
        else:
            self._calibration_state = CalibrationState.ACCUMULATING_CPS

    def _compare_shape_cps(self):
        # self._last_frame
        # self._current_frame
        # get_good_features_to_track
        # compare to last good_features_to_track
        # mostly the same?
        # if so, return True
        # else return False
        return False


