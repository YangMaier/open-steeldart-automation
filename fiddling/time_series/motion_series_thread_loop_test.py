import keyboard
import time

from data_structures.cam_thread import CamThread


def main_loop(preview_name, cam_id: int):
    while True:
        # Your loop logic here
        print(f"{preview_name} working...")

        # Check for 'q' key press to break the loop
        if keyboard.is_pressed('q'):
            print(f"Loop for {preview_name} terminated by user.")
            break

        # Sleep for 1 second
        time.sleep(1)


def start_webcam_threads(webcam_ids):
    for cam_id in webcam_ids:
        thread = CamThread("Camera " + str(cam_id), cam_id, main_loop)
        thread.start()


if __name__ == "__main__":
    start_webcam_threads([0, 1, 2])
