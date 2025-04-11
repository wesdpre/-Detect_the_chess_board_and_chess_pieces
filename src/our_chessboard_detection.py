import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Params
margin = 28
side = 400
square_size = (side - 2 * margin) // 8

import cv2
import numpy as np
import matplotlib.pyplot as plt


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


def rotate_and_crop(image=None, square=None, show=False):
    # Order the 4 points: top-left, top-right, bottom-right, bottom-left
    pts = square.reshape(4, 2)

    # Define destination points for the warp
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

    # Compute the transform matrix and warp
    M = cv2.getPerspectiveTransform(np.float32(pts), dst)
    warped = cv2.warpPerspective(image['original_image'], M, (side, side))

    # Display the warped image
    if show:
        fig, axes = plt.subplots(1, 2, figsize=(10, 6))

        axes[0].imshow(cv2.cvtColor(image['original_image'], cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        axes[1].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Warped Image')
        axes[1].axis('off')
        plt.show()

    return warped, M

def count_black_pixels(image, corner, radius):
    """
    Count the number of black pixels in a given radius around a specified corner.
    """
    x, y = corner
    # Crop a region around the corner within the radius
    x_start = max(0, x - radius)
    y_start = max(0, y - radius)
    x_end = min(image.shape[1], x + radius)
    y_end = min(image.shape[0], y + radius)

    roi = image[y_start:y_end, x_start:x_end]
    black_pixels = np.sum(roi < 50)  # Threshold to detect black pixels (near zero intensity)
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
    return rotated_image

def align_board(image, radius=20, angle_step=1):
    """
    Rotate the image to find the best angle where the bottom-left corner contains the most black pixels.
    """
    max_black_pixels = 0
    best_angle = 0

    corner = (0,0)

    # Get the center of the image

    # Rotate the image from 0 to 360 degrees in small steps
    for angle in range(0, 360, angle_step):
        rotated_image = rotate_board(image, angle)
        black_pixels = count_black_pixels(rotated_image, corner, radius)
        if black_pixels > max_black_pixels:
            max_black_pixels = black_pixels
            best_angle = angle
    
    # Rotate the image to the best angle found
    final_rotated_image = rotate_board(image, best_angle)
    return final_rotated_image, best_angle

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


def display_chessboard_squares(warped):

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
        
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()

    return squares


if __name__ == "__main__":
    # Load the image

    image = cv2.imread('/Users/santiagoromero/Documents/cv/-Detect_the_chess_board_and_chess_pieces/images/G078_IMG092.jpg')



    # Apply filters
    filtered_images = apply_filters(image, show=False)

    # Get chessboard contour
    chess_contour = get_contours(filtered_images, show=True,  kernel_size=(25,25) ,  kernel_usage=True, iterations=4)


    # Rotate and crop the image
    warped_image = rotate_and_crop(filtered_images, chess_contour[0][1], show=False)

    chesboard_grids(warped_image, show = True)





    
