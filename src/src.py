import cv2

def show_imags(img_array, img_display_size=(640, 480)):
    i = 0
    for img in img_array:
        img_name = 'image_' + str(i)
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
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Error: Could not read image {path}")
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        images.append(img)
        gray_images.append(gray_img)
    return images , gray_images