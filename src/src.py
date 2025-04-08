import cv2
import numpy as np
import os

def show_images(img_array, img_names, img_display_size=(640, 480)):
    i = 0
    for img in img_array:
        name = img_names[i]
        img_name = 'image_' + str(i) + '_' + name
        if img is None:
            print(f"Error: Could not read image {img_name}")
            continue
        # Resize the image to a fixed size (optional)
        img = cv2.resize(img, img_display_size)
        cv2.imshow(img_name, img)
        i += 1
    
    # Wait for a key press and close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

def read_images(image_paths):
    images = []
    gray_images = []
    images_names = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Error: Could not read image {path}")
            continue
        
        #img = cv2.cvtColor(img, cv2.COLOR_RG«)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        images.append(img)
        images_names.append(get_filename_from_path(path))
        gray_images.append(gray_img)
    return images, gray_images, images_names

def get_filename_from_path(file_path):
    """
    Extracts the filename (without extension) from a file path.

    Args:
        file_path (str): The full path to the file.

    Returns:
        str: The filename without the extension, or None if there is an error.
    """
    try:
        base_filename = os.path.basename(file_path)  # Get filename with extension
        filename_without_extension = os.path.splitext(base_filename)[0] #split the extension and take the first element.
        return filename_without_extension
    except Exception as e:
        print(f"Error getting filename: {e}")
        return None

def detect_chessboard_corners(images, img_names, resized_width, resized_height, show=True, chessboard_size = (7, 7)):
    
    img_show = []
    ret_array = []
    corners_array = []
    new_image_names = []
    i = 0
    fail_detection = 0
    for img in images:
        if img is None:
            print("Error: Could not read image")
            continue
        name = img_names[i]
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCornersSB(img, chessboard_size, None)
        
        ret_array.append(ret)
        corners_array.append(corners)
        if show:
            if ret:
                original_height, original_width = img.shape[:2]

                # Resize the image to a fixed size (optional)
                img_resize = cv2.resize(img, (resized_width, resized_height))

                # Scale the corner coordinates
                scale_x = resized_width / original_width
                scale_y = resized_height / original_height
                scaled_corners = corners * np.array([[[scale_x, scale_y]]], dtype=np.float32)

                # Draw and display the corners
                draw_img = cv2.drawChessboardCorners(img_resize, chessboard_size, scaled_corners, ret)
                img_show.append(draw_img)
                new_image_names.append(name)
                print(f"Image {name} has corners")
            else:
                print(f"Image {name} does not have corners")
                fail_detection += 1
        i += 1
    print(f"Number of images without corners: {fail_detection}")
    if show:
        show_imags(img_show, new_image_names)
    return ret_array, corners_array
