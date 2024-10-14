import copy
import logging
import pathlib
from typing import List

import numpy as np
import cv2 as cv
import skimage
from skimage.morphology import skeletonize

from data_structures.board_segment import BoardSegment
from data_structures.coordinates import Coordinate2d
from data_structures.hsv_mask_presets import hsv_mask_silver
from data_structures.letters import LetterToDetermine, PreSavedLetter, LetterMatches, PreSavedLetterImages
from image_masking import get_masked_img_by_hsv_values
from operations_on.contours import filter_contour_min_rotated_extent, filter_contour_max_rotated_extent, match_contour, \
    is_elongated
from operations_on.coordinates import get_line_extension_end_point
from operations_on.images import get_img_diff_skikit_similarity
from preset_circled_board_coordinates2 import __DartboardDefinition


def read_numbers_and_add_numbers_to_double_segments(img: np.ndarray, img_height: int, img_width: int, board_center_coordinate: Coordinate2d, double_segments: List[BoardSegment]):
    img_mask_thresh_contour = []
    number_images = []
    double_segments.sort(key=lambda ds: ds.center_cad.angle)

    for double_segment in double_segments:
        double_min_coordinate: Coordinate2d = get_line_extension_end_point(board_center_coordinate, double_segment.low_angle_side_cad.coordinate, 1.06)
        double_max_coordinate: Coordinate2d = get_line_extension_end_point(board_center_coordinate, double_segment.high_angle_side_cad.coordinate, 1.06)
        behind_min_coordinate = get_line_extension_end_point(board_center_coordinate, double_min_coordinate, 1.38)
        behind_max_coordinate = get_line_extension_end_point(board_center_coordinate, double_max_coordinate, 1.38)
        if not \
                0 < behind_min_coordinate.x < img_height or not \
                0 < behind_min_coordinate.y < img_width or not \
                0 < behind_max_coordinate.x < img_height or not \
                0 < behind_max_coordinate.y < img_width:
            img_mask_thresh_contour.append([None, None, None])
            number_images.append(None)
        else:
            pts_warp = np.float32([[p.x, p.y] for p in [double_min_coordinate, behind_min_coordinate, behind_max_coordinate, double_max_coordinate]])
            pts = np.asarray([[p.x, p.y] for p in [double_min_coordinate, behind_min_coordinate, behind_max_coordinate, double_max_coordinate]], dtype=np.int32)
            pts2 = np.float32([[0, 300], [0, 0], [300, 0], [300, 300]])
            M = cv.getPerspectiveTransform(pts_warp, pts2)
            img_copy = copy.deepcopy(img)
            number_img = cv.warpPerspective(img_copy, M, (300, 300))
            number_images.append(number_img)
            mask = np.zeros_like(img)
            cv.fillPoly(mask, [pts], (255, 255, 255))
            masked_img = cv.bitwise_and(img_copy, mask)
            mask_area = cv.contourArea(np.asarray([pts]))

            thresh_img, number_contour = get_number_contour(
                masked_img,
                img_height,
                img_width,
                mask_area
            )
            img_mask_thresh_contour.append([masked_img, thresh_img, number_contour])
            # image_number_contours.append(number_contour)

    letter_sequence = match_letter_images(number_images)

    # live_contours = [LetterToDetermine(mtc[1], mtc[2]) for mtc in img_mask_thresh_contour]
    # letter_sequence = match_letter_contours(live_contours)
    for (double_segment, number) in zip(double_segments, letter_sequence):
        double_segment.number = number

    return double_segments


def get_number_contour(img_letter, img_width, img_height, mask_area: float = 0):
    """ Extracts the contour of the letter

    Applies image filters and tries to extract the contour of the letter

    Args:
        mask_area:
        img_height:
        img_width:
        img_letter: a masked version of the original image frame, containing only the letter and surrounding background

    Returns:
        Either the contour of the letter or None
        Results may vary because the contour filters are not perfect

    """
    if img_height == 1080 and img_width == 1920:
        contour_min_area = 1_000
        contour_max_area = 10_000
    elif img_width == 300 and img_height == 300:
        contour_min_area = 30
        contour_max_area = 500
    else:
        contour_min_area = 200
        contour_max_area = 650
    if mask_area == 0:
        mask_area = contour_max_area * 6

    blur = cv.blur(img_letter, (3, 3))
    grey = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(grey, 80, 255, 0)
    contours_thresh = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[0]
    contours_thresh_filtered = [cnt for cnt in contours_thresh if contour_min_area < cv.contourArea(cnt) < contour_max_area]
    contours_thresh_filtered = [cnt for cnt in contours_thresh_filtered if cv.contourArea(cnt) < mask_area / 6]
    img_thresh = np.zeros([img_height, img_width], np.uint8)  # debug
    cv.drawContours(img_thresh, contours_thresh_filtered, -1, (255, 255, 255), 1)  # debug
    contours_thresh_filtered = [ctf for ctf in contours_thresh_filtered if filter_contour_min_rotated_extent(ctf, 0.1)]
    contours_thresh_filtered = [ctf for ctf in contours_thresh_filtered if filter_contour_max_rotated_extent(ctf, 0.4)]
    contours_thresh_filtered = [ctf for ctf in contours_thresh_filtered if not is_elongated(ctf, 3)]
    img_thresh2 = np.zeros([img_height, img_width], np.uint8)  # debug
    cv.drawContours(img_thresh2, contours_thresh_filtered, -1, (255, 255, 255), -1)  # debug

    # AND multiple solutions
    hsv_filtered = get_masked_img_by_hsv_values(img_letter, hsv_mask_silver)
    laplacian = cv.Laplacian(grey, 0, 1, 5, 3, 1000)
    laplacian_inverted = cv.bitwise_not(laplacian)
    thresh_laplace_inv = cv.threshold(laplacian_inverted, 254, 255, cv.THRESH_OTSU)[1]
    filtered_img = cv.bitwise_and(hsv_filtered, thresh_laplace_inv)

    # try gaussian filter maybe another day, is capable of better results but takes too much time to figure out
    # thresh_gaussian = cv.adaptiveThreshold(grey, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 5, 7)
    # tresh_gaussian_inverted =
    # eroded = cv.erode(th_gaussian2, (3, 3))
    # contours_gaussian, _ = cv.findContours(eroded, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    # contours_filtered_gaussian = [cnt for cnt in contours_gaussian if 200 < cv.contourArea(cnt) < 400]
    # # contours_filtered = [cnt for cnt in contours_filtered if filter_contour_extent(cnt, 0.2)]
    # img_filtered_gaussian = np.zeros([img_height, img_width], np.uint8)
    # cv.drawContours(img_filtered_gaussian, contours_filtered_gaussian, -1, 255, 1)

    if len(contours_thresh_filtered) == 1:
        return img_thresh2, contours_thresh_filtered[0]
    else:
        return img_thresh, None


def match_letter_images(number_images):
    letter_sequence = __DartboardDefinition.letter_sequence_clockwise  # starts with 10, but does not really matter
    letter_clockwise_combinations: List[List[int]] = []
    for i in range(20):
        letter_clockwise_combinations.append(letter_sequence[i:] + letter_sequence[:i])

    # img_thresh_contour = []
    # for img in number_images:
    #     if img is None:
    #         img_thresh_contour.append((None, None, None))
    #     else:
    #         grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #         _, thresh = cv.threshold(grey, 80, 255, 0)
    #         img_thresh_contour.append((img, thresh, None))

    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Jimbob\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
    texts = []
    skeletonized_images = []
    eroded_images = []
    dilated_images = []
    masked_thresh_images = []
    for img in number_images:
        if img is not None:
            grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            _, thresh = cv.threshold(grey, 80, 255, 0)
            test_img_ski = skimage.util.img_as_float(thresh)
            skeleton_img_ski = skeletonize(test_img_ski)
            skeleton_img_cv = skimage.util.img_as_ubyte(skeleton_img_ski)
            # eroded = cv.erode(thresh, np.ones((15, 1), np.uint8))
            # dilated = cv.dilate(eroded, np.ones((15, 15), np.uint8))
            # thresh_masked = cv.bitwise_and(thresh, thresh, mask=dilated)
            text = pytesseract.image_to_string(skeleton_img_cv, lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
            texts.append(text)
            skeletonized_images.append(skeleton_img_cv)
            # eroded_images.append(eroded)
            # dilated_images.append(dilated)
            # masked_thresh_images.append(thresh_masked)
        else:
            texts.append(None)

    letter_image_path_parent = pathlib.Path(__file__).parent.parent.joinpath("media/letter_images")
    pre_saved_letter_folders = []
    for i in letter_sequence:
        pre_saved_letter_images = PreSavedLetterImages(i, [])
        letter_folder_path = letter_image_path_parent.joinpath(str(i))
        if letter_folder_path.exists():
            if letter_folder_path.is_dir():
                for image_path in letter_folder_path.iterdir():
                    if image_path.is_file():
                        img = cv.imread(str(image_path))
                        pre_saved_letter_images.images.append(img)
        pre_saved_letter_folders.append(pre_saved_letter_images)

    letter_sequence_diffs = []
    for letter_sequence in letter_clockwise_combinations:
        # match all presaved letter images, save best match number
        cumulated_diffs = 0
        for letter, live_img in zip(letter_sequence, number_images):
            if live_img is None:
                min_diff = 5000
            else:
                pre_saved_letter_images: PreSavedLetterImages = [pslf for pslf in pre_saved_letter_folders if pslf.number == letter][0]
                if len(pre_saved_letter_images.images) == 0:
                    min_diff = 5000
                else:
                    min_diff = 5000
                    for pre_saved_img in pre_saved_letter_images.images:
                        img_diff = get_img_diff_skikit_similarity(live_img, pre_saved_img)
                        diff = np.int32(np.sum(img_diff) / (img_diff.size / 2))
                        if diff < min_diff:
                            min_diff = diff
                        if diff < 5:
                            print("Found perfect letter match.")
                            return letter_sequence
            # cumulate diffs for the complete letter sequence
            cumulated_diffs += min_diff

        letter_sequence_diffs.append(cumulated_diffs)

    min_sequence_diff = min(letter_sequence_diffs)
    minimized_letter_sequences = [sequence_diff - min_sequence_diff for sequence_diff in letter_sequence_diffs]

    best_sequence_index = np.argmin(minimized_letter_sequences)
    best_sequence = letter_clockwise_combinations[best_sequence_index]

    return best_sequence


def match_letter_contours(frame_letters: list):
    """ Matches live contours with pre saved letter images

    This function
    - reads letter template images from disk
    - extracts the contours from the letter images
    - makes a list with all the extracted letter contours from disk
    - cycles this list 19 more times in order to be able to match the live contours with every possible board rotation
    - matches the live contours with every possible board rotation/cycle
    - gets a matching score for each rotation
    - checks which rotation is the best match (lowest match score)

    Args:
        frame_letters: extracted from live image, filled with None for letters that couldn't be extracted

    Returns:
        the best rotation match in the form of an int sequence

    Assumptions:
        - live contours are in clockwise order

    Notes:
        nearly every board has some letters that have background noise and will not be useful for matching
        these letters should not be pre saved as they are really hard to extract
        maybe ~12 letter images will be pre saved as templates
        the same is true for the letters contours that are extracted live
        maybe 4-5 letters can be read from the live image, to a max of 12 if the whole number ring is visible
        that approximates to 5 * 12 * 20 = 1200 contour matchings instead of 20 * 20 * 20 = 8000

        currently cv.matchShapes is used for contour comparison. It is not very robust.
        That's why every possible letter will be matched and a matching score for each rotation will be calculated.
        This way its ensured that the best match will be found

    Possible improvements:
        - use a letter matching algorithm that is more robust, cv.matchShapes seems to have some issues
        - if that matching algorithm is robust enough, the rotations will not be necessary and can be removed
          that will improve the runtime massively because the mean comparisons made should be 1 to maybe 3

    """
    # it's necessary to keep the clockwise letter sorting intact
    # letter contours that could not be extracted from live image should be None type
    # the function requires None type frame_letters as fill values for letters that are not visible
    assert len(frame_letters) == 20, "Expected len 20 of letter contours list, got " + str(len(frame_letters))

    letter_sequence_clockwise = [1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5, 20]
    letter_image_path = pathlib.Path(__file__).parent.parent.joinpath("media/letter_images_old")

    letter_contours = dict()
    for i in range(1, 21):
        letter_img_file_path = letter_image_path.joinpath(f"{i}.png")
        if letter_img_file_path.is_file():  # checks if file exists
            letter_img = cv.imread(str(letter_img_file_path))
            thresh, number_contour = get_number_contour(letter_img, 1920, 1080)
            if number_contour is not None:
                # letter_img = np.zeros([480, 640], np.uint8)  # currently wrong size
                # cv.drawContours(letter_img, contours_thresh_filtered, -1, 255, 1)  # debug
                letter_contours[i] = PreSavedLetter(i, thresh, number_contour)
            else:
                logging.warning(f"Could not extract contour for letter {i}")
                letter_contours[i] = PreSavedLetter(i, thresh, None)
        else:
            logging.debug(f"Image for letter {i} not found.")
            letter_contours[i] = PreSavedLetter(i, None, None)

    letter_clockwise_combinations = []
    for i in range(20):
        letter_clockwise_combinations.append(letter_sequence_clockwise[i:] + letter_sequence_clockwise[:i])

    letter_sequences = []
    for lcc in letter_clockwise_combinations:
        sequence: [PreSavedLetter] = [letter_contours.get(number) for number in lcc]
        letter_sequences.append(sequence)

    matches: [[LetterMatches]] = []
    for sequence in letter_sequences:
        sequence_matches = []
        for (frame_letter, pre_saved_letter) in zip(frame_letters, sequence):
            if frame_letter.contour is not None and pre_saved_letter.contour is None:
                # there are two possible reasons for this case:
                # - this rotation is just not viable because there is no pre saved letter for an easily readable letter
                # - the live letter is a falsy extraction, but then every rotation will get the same penalty
                sequence_matches.append(
                    LetterMatches(
                        frame_letter,
                        pre_saved_letter,
                        1000  # big matching score penalty
                    )
                )
            else:
                sequence_matches.append(
                    LetterMatches(
                        frame_letter,
                        pre_saved_letter,
                        match_contour(frame_letter.contour, pre_saved_letter.contour)  # the important part happens here
                    )
                )
        matches.append(sequence_matches)

    # the number of comparisons per sequence does not change, so we can add up the match values for each sequence
    # and use the lowest result as the overall best match
    match_results = [sum([lm.match for lm in m if lm.match is not None]) for m in matches]
    min_match_index = np.argmin(match_results)
    best_combination = letter_clockwise_combinations[min_match_index]

    return best_combination
