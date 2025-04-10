import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

side = 400

def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def apply_filters(image, show = False):
    # Convert to grayscale
    gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # gaussian blur
    # Apply Gaussian blur
    # to reduce noise and improve corner detection
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    # Apply Canny filter
    image_canny = cv2.Canny(gray_image, 50, 200)

    image_dark = adjust_gamma(image, 0.2)
    gray_dark = cv2.cvtColor(image_dark, cv2.COLOR_BGR2GRAY)
    blurred_dark = cv2.GaussianBlur(gray_dark, (11,11), 0)
    image_canny_dark = cv2.Canny(blurred_dark, 0, 140)

    if show:
        # Display images
        plt.subplot(221), plt.imshow(image), plt.title('Original Image')
        plt.subplot(222), plt.imshow(image_canny, cmap='gray'), plt.title('Canny Image')
        plt.subplot(223), plt.imshow(gray_dark, cmap='gray'), plt.title('Dark Gray Image')
        plt.subplot(224), plt.imshow(image_canny_dark, cmap='gray'), plt.title('Dark Canny Image')
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

def rotate_and_crop(image = None, square = None, show = False):
    # Order the 4 points: top-left, top-right, bottom-right, bottom-left
    pts = square.reshape(4, 2)

    # Define destination points for the warp
    
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

    # Compute the transform matrix and warp
    M = cv2.getPerspectiveTransform(np.float32(pts), dst)
    warped = cv2.flip(cv2.warpPerspective(image['original_image'], M, (side, side)), 1) 

    # Display the warped image
    if show:
        plt.subplot(121), plt.imshow(image['original_image']), plt.title('Original Image')
        plt.subplot(122), plt.imshow(warped), plt.title('Warped Image')
        plt.show()
    
    return warped


def chesboard_grids(warped_image, show = False):
    margin = 28
    square_size = (side-2*margin)//8
    img_copy = warped_image.copy()
    for i in range(9):
        x = margin + i * square_size
        cv2.line(img_copy, (x, margin), (x, margin + 8 * square_size), (0, 255, 0), 2)

    # Draw horizontal lines
    for i in range(9):
        y = margin + i * square_size
        cv2.line(img_copy, (margin, y), (margin + 8 * square_size, y), (0, 255, 0), 2)

    if show:
        plt.subplot(121), plt.imshow(img_copy), plt.title('image with grid')
        plt.show()



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





    
