import cv2
import numpy as np
from collections import deque

def nothing(x):
    pass

def initialize_background(camera, delay=30):
    """
    Initialize the background by capturing multiple frames and averaging them.
    This helps in reducing noise and getting a stable background.
    """
    print("Initializing background. Please ensure the dartboard is free of darts.")
    frames = []
    for i in range(delay):
        ret, frame = camera.read()
        if not ret:
            continue
        frames.append(frame)
        cv2.imshow("Initializing Background", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if not frames:
        print("No frames captured for background initialization.")
        return None
    background = np.median(frames, axis=0).astype(dtype=np.uint8)
    cv2.destroyWindow("Initializing Background")
    return background

def compute_difference(background: np.ndarray, current_frame: np.ndarray, color_space: int):
    """
    Compute absolute differences between the background and the current frame
    across different color spaces and return the results.
    """
    bg_color_space = cv2.cvtColor(background, color_space)
    curr_color_space = cv2.cvtColor(current_frame, color_space)
    diff_color_space = cv2.absdiff(bg_color_space, curr_color_space)
    return diff_color_space

def process_frame(frame, differences, blur_size=5, threshold_value=30, min_contour_area=300):
    """
    Process the differences to detect darts, combining results from all color spaces,
    and overlay the detection on the original frame.
    """
    combined_mask = np.zeros_like(differences[0])
    dart_tip_positions = []

    for diff in differences:
        # Apply Gaussian Blur to reduce noise
        blur_size = max(1, blur_size)
        if blur_size % 2 == 0:
            blur_size += 1
        diff_blur = cv2.GaussianBlur(diff, (blur_size, blur_size), 0)

        # Apply thresholding
        if len(diff_blur.shape) == 2:  # Grayscale
            _, thresh = cv2.threshold(diff_blur, threshold_value, 255, cv2.THRESH_BINARY)
        else:  # Multi-channel images (Lab, YCrCb)
            thresh = cv2.threshold(np.sum(diff_blur, axis=2).astype(np.uint8), threshold_value, 255, cv2.THRESH_BINARY)[1]

        # Combine masks
        combined_mask = cv2.bitwise_or(combined_mask, thresh)

    # Dilate the combined mask
    combined_mask = cv2.dilate(combined_mask, None, iterations=2)

    # Find contours in the combined mask
    contours, _ = cv2.findContours(combined_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the original frame for highlighting
    highlighted = frame.copy()

    for contour in contours:
        if cv2.contourArea(contour) < min_contour_area:
            continue

        # Draw the contour
        cv2.drawContours(highlighted, [contour], 0, (0, 255, 0), 2)

        # Find the bottom-most point of the contour
        bottom_most_point = tuple(contour[contour[:, :, 1].argmax()][0])
        dart_tip_positions.append(bottom_most_point)

        # Draw the tip on the highlighted image
        cv2.circle(highlighted, bottom_most_point, 5, (255, 0, 0), -1)  # Blue dot for the tip
        cv2.putText(highlighted, f"({bottom_most_point[0]}, {bottom_most_point[1]})", 
                    (bottom_most_point[0] + 10, bottom_most_point[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return highlighted, dart_tip_positions

def main():
    # Initialize video capture
    camera = cv2.VideoCapture(0)  # Change the index if you have multiple cameras

    if not camera.isOpened():
        print("Cannot open camera")
        return

    # Set camera resolution to 1280x720
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Initialize background
    background = initialize_background(camera)
    if background is None:
        print("Background initialization failed.")
        camera.release()
        return

    print("Background initialized. Starting live detection.")
    print("Press 'q' to quit.")
    print("Press 'r' to reinitialize background.")

    # Create trackbars for adjusting Gaussian blur size and threshold value
    cv2.namedWindow("Settings")
    cv2.createTrackbar("Blur Size", "Settings", 7, 50, nothing)
    cv2.createTrackbar("Threshold", "Settings", 25, 255, nothing)

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Get trackbar positions
        blur_size = cv2.getTrackbarPos("Blur Size", "Settings")
        threshold_value = cv2.getTrackbarPos("Threshold", "Settings")

        # Compute differences across different color spaces
        diff_gray = compute_difference(background, frame, cv2.COLOR_BGR2GRAY)
        diff_lab = compute_difference(background, frame, cv2.COLOR_BGR2Lab)
        diff_ycrcb = compute_difference(background, frame, cv2.COLOR_BGR2YCrCb)

        cv2.imshow("diff_gray", diff_gray)
        cv2.imshow("diff_lab", diff_lab)
        cv2.imshow("diff_ycrcb", diff_ycrcb)

        differences = [diff_gray, diff_lab, diff_ycrcb]
        
        # Process the frame
        highlighted, dart_tip_positions = process_frame(frame, differences, blur_size, threshold_value)

        # Display the processed frame
        cv2.imshow("Dart Detection - Overlay", highlighted)

        # Print detected dart tips
        if dart_tip_positions:
            print(f"Detected Tips: {dart_tip_positions}")

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reinitialize background if needed
            background = initialize_background(camera)
            if background is None:
                print("Background reinitialization failed.")
                break

    # Release resources
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    