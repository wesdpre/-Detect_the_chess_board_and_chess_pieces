import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def homography_transformation(img1, img2, show = False):
    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # Apply FLAN Matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.match(des1, des2)

    # Obtain points corresponding to the matches in the query and train images
    query_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    train_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Obtain homography that represents the transformation from the points of the train image into the position of the query image
    M, mask = cv2.findHomography(train_pts, query_pts, cv2.RANSAC, 5.0)

    # Filter the inliers from the mask
    mask = mask.ravel().astype(bool)
    filtered_matches = tuple(matches[i] for i in range(len(matches)) if i in np.where(mask)[0])

    match_output = cv2.drawMatches(img1, kp1, img2, kp2, filtered_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Draw blue lines
    #img2_lines = cv2.polylines(img2_bgr, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)
    
    if show:
        cv2.imshow('Query', img1)
        cv2.imshow('Original Image', img2)
        cv2.imshow('Feature Matching Image', match_output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

def background_subtraction(image, square, show = False):
    # Create a mask with the same dimensions as the image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Fill the polygon on the mask
    cv2.fillPoly(mask, [square], 255)

    masked_image = cv2.bitwise_and(image, image, mask=mask)
    if show:
        # Display the images
        cv2.imshow("Original Image", image)
        cv2.imshow("Masked Image", masked_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return masked_image




def crop_chessboard(image, square, show = False, save= False):


    p1 = square[0][0]
    p2 = square[1][0]
    angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / np.pi

    # Calculate the image center
    image_center = tuple(np.array(image.shape[1::-1]) / 2)

    # Create the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    # Perform the rotation on the entire image
    rotated_image = cv2.warpAffine(image, rotation_matrix, image.shape[1::-1])

    # Convert square coordinates to homogeneous coordinates
    square_homogeneous = np.concatenate((square.reshape(-1, 2), np.ones((4, 1))), axis=1)

    # Apply the rotation matrix to the square's coordinates
    rotated_square_homogeneous = np.dot(square_homogeneous, rotation_matrix.T)

    # Convert back to regular coordinates
    rotated_square = rotated_square_homogeneous[:, :2].reshape(4, 1, 2).astype(np.int32)

    # Find the bounding rectangle of the rotated square
    rect = cv2.boundingRect(rotated_square)
    x, y, w, h = rect

    # Crop the region of interest (ROI) from the rotated image
    cropped_rotated_image = rotated_image[y:y + h, x:x + w]

    if save:
        cv2.imwrite('cropped_rotated_image.jpg', cropped_rotated_image)

    if show:
        # Display the images
        cv2.imshow("Rotated Image", rotated_image)
        cv2.imshow("Cropped Rotated Image", cropped_rotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return cropped_rotated_image, rotated_square

def getting_contours(image, show = False):

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get the dimensions of the image
    height, width, channels = image.shape

    print(f"Image Width: {width}, Image Height: {height}, Channels: {channels}")

    # Preprocessing

    edges = cv2.Canny(gray, 50, 200)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Optional: draw all contours to debug
    image_with_contours = image.copy()
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 3)


    squares = []
    for cnt in contours:
        epsilon = 0.05 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(cnt)
            if area > 500:  # This is the area threshold of ouw chessboard
                squares.append(approx)

    print(f"Number of squares detected: {len(squares)}")
    print(f"Squares: {squares}")

    if show:
        
        image_detected = image.copy()
        cv2.drawContours(image_detected, squares, -1, (0, 0, 255), 10)
        # Show result

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))
        plt.title("All Contours")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(image_detected, cv2.COLOR_BGR2RGB))
        plt.title("Detected Squares")
        plt.axis("off")

        plt.subplot(1, 2, 1)
        plt.imshow(edges)
        plt.title("edges")
        plt.axis("off")
        plt.show()

    return squares[0]

if __name__ == "__main__":
    # Getting contours



    dataDir = 'images'
    image = cv2.imread(os.path.join(dataDir, "42", 'G042_IMG086.jpg'))# Change this, according to your image's path
    
    imgage = cv2.imread(os.path.join(dataDir, "42", 'G042_IMG086.jpg'))
    gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # Apply Canny filter
    image_canny = cv2.Canny(imgage, 50, 200)

    # Create RGB copy of image
    rgb_image = cv2.cvtColor(imgage, cv2.COLOR_BGR2RGB)

    square = getting_contours(image, show=True)

    # Assuming we want to crop the first detected square

    #image2 = background_subtraction(image2, square, show=False)

    # this saves the image as cropped_rotated_image.jpg
    #cropped_rotated_image, rotated_square = crop_chessboard(image2, square, show=False)
    #homography_transformation(image, cropped_rotated_image, show=True)
