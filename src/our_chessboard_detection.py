import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Params
margin = 50
side = 512
square_size = (side - 2 * margin) // 8

def read_image(image_path, show=False):
    # Read the image
    image = cv2.imread(image_path)

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if show:
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.show()

    return image


def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


def apply_filters(image, show=False):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian blur
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply Canny filter
    image_canny = cv2.Canny(gray_image, 50, 200)

    # Darken the image
    image_dark = adjust_gamma(image, 0.2)
    gray_dark = cv2.cvtColor(image_dark, cv2.COLOR_BGR2GRAY)
    blurred_dark = cv2.GaussianBlur(gray_dark, (11, 11), 0)
    image_canny_dark = cv2.Canny(blurred_dark, 0, 140)

    if show:

        # Create subplots and disable axes individually
        fig, axes = plt.subplots(1, 4, figsize=(20, 8))
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')  # Turn off axis for this subplot
        axes[1].imshow(cv2.cvtColor(image_canny, cv2.COLOR_GRAY2RGB))
        axes[1].set_title('Canny Image')
        axes[1].axis('off')  # Turn off axis for this subplot
        axes[2].imshow(cv2.cvtColor(gray_dark, cv2.COLOR_GRAY2RGB))
        axes[2].set_title('Dark Gray Image')
        axes[2].axis('off')  # Turn off axis for this subplot
        axes[3].imshow(cv2.cvtColor(image_canny_dark, cv2.COLOR_GRAY2RGB))
        axes[3].set_title('Dark Canny Image')
        axes[3].axis('off')  # Turn off axis for this subplot
        # Adjust space between subplots
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        # Show the plot
        plt.show()

    return {
        'original_image': image,
        'gray_image': gray_image,
        'blurred': blurred,
        'image_canny': image_canny,
        'image_dark': image_dark,
        'gray_dark': gray_dark,
        'blurred_dark': blurred_dark,
        'image_canny_dark': image_canny_dark
    }


def get_contours(image = {}, show = False, kernel_size = (5, 5), kernel_usage = False, iterations = 1):


    contours, _ = cv2.findContours(image['image_canny_dark'], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if kernel_usage:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        closed = cv2.morphologyEx(image['image_canny_dark'], cv2.MORPH_CLOSE, kernel, iterations = iterations)
        contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Optional: draw all contours to debug
    image_with_contours = image['original_image'].copy()
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 4)


    squares = []
    max_area = 0
    major_approx = None

    for cnt in contours:
        epsilon = 0.05 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(cnt)
            squares.append((area, approx))
            if area > max_area:
                max_area = area
                major_approx = approx

    # Sort all squares by area (largest to smallest)
    squares = sorted(squares, key=lambda x: x[0], reverse=True)

    if show:
        # Draw the detected square
        image_detected = image['original_image'].copy()
        cv2.drawContours(image_detected, [major_approx], -1, (0, 0, 255), 10)
        # Show result
        plt.figure(figsize=(10, 5))
        plt.subplot(121), plt.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB)), plt.title("All Contours")
        plt.axis("off")

        plt.subplot(122), plt.imshow(cv2.cvtColor(image_detected, cv2.COLOR_BGR2RGB)), plt.title("Detected Square")
        plt.axis("off")
        plt.show()

    return squares


def order_points(pts):
    """
    Orders 4 points in the order: top-left, top-right, bottom-right, bottom-left
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]        # Top-left
    rect[2] = pts[np.argmax(s)]        # Bottom-right
    rect[1] = pts[np.argmin(diff)]     # Top-right
    rect[3] = pts[np.argmax(diff)]     # Bottom-left

    return rect

def rotate_and_crop(image=None, square=None, side=512, show=False):
    """
    Applies a perspective warp to the region defined by `square` in the `image`.
    
    Parameters:
    - image: dictionary with key 'original_image' containing the BGR image (np.ndarray)
    - square: array-like of 4 (x, y) points
    - side: output size in pixels (square)
    - show: if True, displays the original and warped images with corner labels
    
    Returns:
    - warped image
    - 3x3 homography matrix used
    """
    # Step 1: Ensure square is reshaped and ordered
    pts = order_points(np.array(square).reshape(4, 2))

    # Step 2: Define the destination points
    dst = np.array([
        [0, 0],
        [side - 1, 0],
        [side - 1, side - 1],
        [0, side - 1]
    ], dtype="float32")

    # Step 3: Compute the homography
    M = cv2.getPerspectiveTransform(pts, dst)

    # Step 4: Warp the image
    warped = cv2.warpPerspective(image['original_image'], M, (side, side))

    # Step 5: Optional visualization
    if show:
        debug_img = image['original_image'].copy()
        for i, (x, y) in enumerate(pts):
            cv2.circle(debug_img, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(debug_img, f'{i}', (int(x), int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original with Points')
        axes[0].axis('off')
        axes[1].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Warped (Rotated + Cropped)')
        axes[1].axis('off')
        plt.tight_layout()
        plt.show()

    return warped, M

def count_black_pixels(image, corner, radius, debug=False):
    """
    Count the number of black pixels in a given radius around a specified corner.
    """
    x, y = corner
    # Crop a region around the corner within the radius
    x_start = max(0, x)
    y_start = max(0, y)
    x_end = min(image.shape[1], x_start + 2*radius)
    y_end = min(image.shape[0], y_start + 2*radius)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    roi = image[y_start:y_end, x_start:x_end]

    black_pixels = np.sum(roi < 150)  # Threshold to detect black pixels (near zero intensity)
    return black_pixels

def rotate_board(image, angle):
    """
    Rotate the image around the center by the specified angle.
    """
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    (h, w) = image.shape[:2]
    # Calculate the new size of the rotated image
    new_w = int(w * np.abs(rotation_matrix[0, 0]) + h * np.abs(rotation_matrix[0, 1]))
    new_h = int(h * np.abs(rotation_matrix[0, 0]) + w * np.abs(rotation_matrix[0, 1]))
    
    # Adjust the rotation matrix to ensure the image fits within the new size
    rotation_matrix[0, 2] += (new_w - w) / 2
    rotation_matrix[1, 2] += (new_h - h) / 2
    
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h))
    return rotated_image, rotation_matrix

def align_board(image, radius=20, angle_step=1, show=False):
    """
    Rotate the image to find the best angle where the bottom-left corner contains the most black pixels.
    Visualizes the region being analyzed during the process.
    """
    max_black_pixels = 0
    best_angle = 0

    corner = (0, 512 - 2 * radius)

    for angle in range(0, 360, angle_step):
        rotated_image, M = rotate_board(image, angle)
        print('Angle')
        print(angle)

        # Get region coordinates
        x, y = corner
        x_start = max(0, x)
        y_start = max(0, y)
        x_end = min(rotated_image.shape[1], x_start + 2*radius)
        y_end = min(rotated_image.shape[0], y_start + 2*radius)

        black_pixels = count_black_pixels(rotated_image, corner, radius)
        print('Px Count')
        print(black_pixels)
        if black_pixels > max_black_pixels:
            max_black_pixels = black_pixels
            best_angle = angle
        
        if show:
            # Draw rectangle showing the region being analyzed
            display_image = rotated_image.copy()
            cv2.rectangle(display_image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

            plt.figure(figsize=(10, 6))
            plt.imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
            plt.title(f"Angle: {angle} degrees")
            plt.axis('off')
            plt.show()

    # Rotate to best angle
    final_rotated_image, M = rotate_board(image, best_angle)

    if show:
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(final_rotated_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Best Angle: {best_angle} degrees")
        plt.axis('off')
        plt.show()

    return final_rotated_image, M

def inverse_rotate_crop(warped=None, M=None, angle=None, original_image=None, show=False):
    """
    Overlays the unwarped content back onto the original image,
    preserving the original context.

    Parameters:
    - warped: The warped image (square cropped and perspective-corrected).
    - M: Perspective matrix used for warping.
    - original_image: The original full image.
    - show: Whether to display the result.

    Returns:
    - result: The original image with the warped content overlaid.
    """

    if M is None or warped is None or original_image is None or angle is None:
        raise ValueError("Missing required parameters: 'warped', 'M', 'original_image' or 'angle'")

    warped = rotate_board(warped, (360-angle) % 360)

    h, w = original_image.shape[:2]
    M_inv = np.linalg.inv(M)

    # Warp the content back to original shape
    unwarped = cv2.warpPerspective(warped, M_inv, (w, h))

    # Create a mask of the unwarped image (non-black area)
    gray_unwarped = cv2.cvtColor(unwarped, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_unwarped, 1, 255, cv2.THRESH_BINARY)

    # Invert mask to get the background of the original image
    mask_inv = cv2.bitwise_not(mask)

    # Convert mask to 3-channel
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_inv_3ch = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)

    # Keep only the original background where unwarped won't be applied
    background = cv2.bitwise_and(original_image, mask_inv_3ch)

    # Keep only the unwarped foreground
    foreground = cv2.bitwise_and(unwarped, mask_3ch)

    # Combine the two
    result = cv2.add(background, foreground)

    if show:
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title("Original Image with Inverse Warped Content Overlaid")
        plt.axis('off')
        plt.show()

    return result


def chesboard_grids(warped_image, show = False):
    img_grid = warped_image.copy()

    for i in range(9):
        x = margin + i * square_size
        cv2.line(img_grid, (x, margin), (x, margin + 8 * square_size), (0, 255, 0), 2)

    # Draw horizontal lines
    for i in range(9):
        y = margin + i * square_size
        cv2.line(img_grid, (margin, y), (margin + 8 * square_size, y), (0, 255, 0), 2)

    if show:
        plt.figure(figsize=(10,5))
        plt.imshow(cv2.cvtColor(img_grid, cv2.COLOR_BGR2RGB))
        plt.title('Image with grid')
        plt.axis('off')
        plt.show()

    return img_grid


def display_chessboard_squares(warped, show=False):

    img_grid = warped.copy()
    squares = []
    # Extract each square from the grid
    for row in range(8):
        for col in range(8):
            # Coordiantes of squares
            x1 = margin + col * square_size
            y1 = margin + row * square_size
            x2 = x1 + square_size
            y2 = y1 + square_size

            # Crop the square from the image
            square = warped[y1:y2, x1:x2]
            # Convert the square from BGR to RGB (for matplotlib)
            square_rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
            # Append the square to the list of squares
            squares.append(square_rgb)

    # Create an 8x8 grid plot using matplotlib
    fig, axes = plt.subplots(8, 8, figsize=(10, 10))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(squares[i])
        ax.axis('off')
        ax.set_title(f"{i}", fontsize=10)
        
    if show:
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.show()

    return squares


def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def adjust_contrast(image, alpha=1.5, beta=0):
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

def normalize_white_pieces(square_rgb):
    hsv = cv2.cvtColor(square_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    mask = (
        (h >= 15) & (h <= 45) &
        (s >= 15) & (s <= 200) &
        (v >= 100) & (v <= 240)
    )
    hsv[mask] = (30, 180, 200)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def display_chessboard_squares(warped, gamma=1.5, show= False):
    img_grid = warped.copy()
    squares = []

    for row in range(8):
        for col in range(8):
            x1 = margin + col * square_size
            y1 = margin + row * square_size
            x2 = x1 + square_size
            y2 = y1 + square_size

            square = warped[y1:y2, x1:x2]
            square_rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
            image_light = adjust_gamma(square_rgb, gamma)
            image_norm = normalize_white_pieces(image_light)
            image_contrast = adjust_contrast(image_norm, alpha=1.3, beta=0)
            image_filtered = cv2.medianBlur(image_contrast, 5)
            squares.append(image_filtered)

    fig, axes = plt.subplots(8, 8, figsize=(10, 10))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(squares[i])
        ax.axis('off')
        ax.set_title(f"{i}", fontsize=10)

    if show:
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.show()

    return squares

def check_piece_at_center(square, black_piece=(45, 45, 45), white_piece=(255, 252, 94), atol=50):
    # Flatten and calculate average color around center for initial classification
    center_x = square.shape[1] // 2
    center_y = square.shape[0] // 2
    crop_size = 3
    crop_x1 = max(center_x - crop_size // 2, 0)
    crop_y1 = max(center_y - crop_size // 2, 0)
    crop_x2 = min(center_x + crop_size // 2 + 1, square.shape[1])
    crop_y2 = min(center_y + crop_size // 2 + 1, square.shape[0])
    cropped_region = square[crop_y1:crop_y2, crop_x1:crop_x2]
    avg_color = np.mean(cropped_region, axis=(0, 1))

    # Determine expected color
    if np.allclose(avg_color, white_piece, atol=atol):
        target_color = white_piece
        label = "WHITE"
    elif np.allclose(avg_color, black_piece, atol=atol):
        target_color = black_piece
        label = "BLACK"
    else:
        return "EMPTY", None

    # Find all pixels in the square close to the target color
    mask = np.all(np.isclose(square, target_color, atol=atol), axis=-1)
    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        return label, None  # No matching pixels found

    xmin = int(np.min(xs))
    xmax = int(np.max(xs))
    ymin = int(np.min(ys))
    ymax = int(np.max(ys))

    return label, (xmin, ymin, xmax, ymax)

def process_chessboard(squares):
    board_matrix = np.zeros((8, 8))
    white_count = 0
    black_count = 0
    piece_coords = []

    for i in range(8):
        for j in range(8):
            square = squares[i * 8 + j]

            # Check the piece at the center of the square and get its local bounding box
            piece_type, local_box = check_piece_at_center(square)

            if piece_type in {"WHITE", "BLACK"}:
                board_matrix[i, j] = 1

                if piece_type == "WHITE":
                    white_count += 1
                elif piece_type == "BLACK":
                    black_count += 1

                if local_box is not None:
                    # Local coordinates within the square (no offset)
                    piece_coords.append({
                        "xmin": int(local_box[0]),
                        "ymin": int(local_box[1]),
                        "xmax": int(local_box[2]),
                        "ymax": int(local_box[3])
                    })

    # Output (optional)
    print("Board Matrix (8x8):")
    # Convert board_matrix into common array
    board_matrix = np.array(board_matrix, dtype=int)
    print(board_matrix)
    print(f"White pieces: {white_count}")
    print(f"Black pieces: {black_count}")
    total_pieces = white_count + black_count
    print(f"Total pieces: {total_pieces}")

    return board_matrix, piece_coords, total_pieces

def reverse_piece_coordinates(piece_coords, rotation_angle, perspective_matrix, rotated_image_shape):
    """
    Reverses perspective and rotation transforms to map piece bounding boxes
    back to coordinates in the original image.

    Args:
        piece_coords: list of dicts with keys 'xmin', 'xmax', 'ymin', 'ymax'
        rotation_angle: angle used to rotate the warped image
        perspective_matrix: M used for perspective warp (cv2.getPerspectiveTransform)
        rotated_image_shape: shape of the rotated (aligned) image

    Returns:
        A list of dictionaries with original image coordinates using the same bounding box keys.
    """
    h_rot, w_rot = rotated_image_shape[:2]
    center = (w_rot // 2, h_rot // 2)

    # Create inverse rotation matrix
    R = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    R_3x3 = np.vstack([R, [0, 0, 1]])
    R_inv = np.linalg.inv(R_3x3)

    # Collect all corner points from bounding boxes
    all_points = []
    box_indices = []

    for idx, box in enumerate(piece_coords):
        corners = [
            [box['xmin'], box['ymin']],
            [box['xmax'], box['ymin']],
            [box['xmax'], box['ymax']],
            [box['xmin'], box['ymax']]
        ]
        all_points.extend(corners)
        box_indices.extend([idx] * 4)

    # Apply inverse rotation
    points_np = np.array(all_points, dtype=np.float32)
    points_hom = np.hstack([points_np, np.ones((len(points_np), 1))])
    coords_in_warped = (R_inv @ points_hom.T).T[:, :2]

    # Apply inverse perspective
    M_inv = np.linalg.inv(perspective_matrix)
    coords_in_warped = np.array(coords_in_warped, dtype=np.float32).reshape(-1, 1, 2)
    coords_in_original = cv2.perspectiveTransform(coords_in_warped, M_inv).reshape(-1, 2)

    # Group back into box dictionaries
    original_boxes = [{} for _ in range(len(piece_coords))]
    for i in range(len(piece_coords)):
        box_points = coords_in_original[i * 4: (i + 1) * 4]
        x_vals = box_points[:, 0]
        y_vals = box_points[:, 1]
        original_boxes[i] = {
            "xmin": int(np.min(x_vals)),
            "ymin": int(np.min(y_vals)),
            "xmax": int(np.max(x_vals)),
            "ymax": int(np.max(y_vals))
        }

    return original_boxes

def offset_piece_coords(piece_coords, board_matrix):
    """
    Converts local square-relative piece bounding boxes to global (warped image) coordinates
    by applying offset based on the square's position on the 8x8 board.

    Args:
        piece_coords: list of dicts with keys 'xmin', 'ymin', 'xmax', 'ymax' (local to square)
        board_matrix: 8x8 matrix with 1s where pieces are detected
        margin: pixel offset around the board in the warped image
        square_size: size of each square in pixels

    Returns:
        List of dicts with global (warped image) coordinates
    """
    global_coords = []
    idx = 0  # index for piece_coords

    for i in range(8):
        for j in range(8):
            if board_matrix[i, j] == 1:
                local_box = piece_coords[idx]
                offset_x = margin + j * square_size
                offset_y = margin + i * square_size

                global_coords.append({
                    "xmin": local_box["xmin"] + offset_x,
                    "ymin": local_box["ymin"] + offset_y,
                    "xmax": local_box["xmax"] + offset_x,
                    "ymax": local_box["ymax"] + offset_y
                })
                idx += 1

    return global_coords

import json

from src.our_chessboard_detection import (read_image, apply_filters, get_contours, 
                                      rotate_and_crop, align_board,chesboard_grids, 
                                      process_chessboard, display_chessboard_squares,
                                      offset_piece_coords, reverse_piece_coordinates)

def process_image(image_path):

    image = read_image(image_path, False)
    filtered_images = apply_filters(image, False)
    chess_contour = get_contours(filtered_images, show=False,  kernel_size=(25,25) ,  kernel_usage=True, iterations=4)
    warped_image, M = rotate_and_crop(filtered_images, chess_contour[0][1], show=False)
    rotated_image, best_angle = align_board(warped_image, radius=12, angle_step=90, show=False)
    squares = display_chessboard_squares(rotated_image, show = False)
    board_matrix, piece_coords, pieces_count = process_chessboard(squares)
    piece_coords_global = offset_piece_coords(piece_coords, board_matrix,)

    # Reverse transforms
    original_coords = reverse_piece_coordinates(
        piece_coords_global,
        rotation_angle=best_angle,  # from align_board
        perspective_matrix=M,       # from rotate_and_crop
        rotated_image_shape=rotated_image.shape
    )
    print(original_coords)


    # Convert the board matrix to a compact nested list format
    board_matrix_list = board_matrix.tolist()

    # Convert original_coords to a JSON-serializable format
    original_coords_list = [dict(coord) for coord in original_coords]

    return {
        "image_path": image_path,
        "num_pieces": pieces_count,
        "board": board_matrix_list,
        "detected_pieces": original_coords_list
    }

def generate_output(input_file_path='input.json', output_file_path='output.json'):
    with open(input_file_path, 'r') as infile:
        data = json.load(infile)

    results = []
    for image_path in data.get('image_files', []):
        result = process_image(image_path)
        results.append(result)

    with open(output_file_path, 'w') as outfile:
        json.dump(results, outfile, cls=CompactBoardJSONEncoder)



import json

class CompactBoardJSONEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        kwargs['indent'] = 4  # Still keep overall formatting
        super().__init__(*args, **kwargs)

    def encode(self, o):
        if isinstance(o, list):
            # Compact encoding for lists of integers
            if all(isinstance(i, list) and all(isinstance(j, int) for j in i) for i in o):
                return "[\n" + ",\n".join(
                    ["    " + json.dumps(row) for row in o]
                ) + "\n]"
        return super().encode(o)


if __name__ == "__main__":
    # Example usage
    generate_output('json_requests/input.json', 'output.json')