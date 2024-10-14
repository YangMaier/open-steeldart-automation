import os
import pathlib
import shutil

# export dart images from three_dart_motion_series folders to dart_images folder
# exports only images with darts in them and no images with motion in them

path_three_dart_motion_series_folder = pathlib.Path("A:/dart_motion_series")
path_flat_dart_images = pathlib.Path("A:/dart_images")
for three_darts_folder in path_three_dart_motion_series_folder.iterdir():
    for cam_folder in three_darts_folder.iterdir():
        if cam_folder.is_dir():
            for folder_or_image in cam_folder.iterdir():
                if folder_or_image.is_dir():
                    all_images_paths = list(folder_or_image.iterdir())
                    last_image_path = all_images_paths[-1]
                    if last_image_path.is_file():
                        destination_path = path_flat_dart_images.joinpath(last_image_path.name)
                        os.makedirs(path_flat_dart_images, exist_ok=True)
                        shutil.copy(last_image_path, destination_path)

print("done copying files to flat structure.")
