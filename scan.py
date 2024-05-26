import cv2
import os
import numpy as np

folder_path = "cube image"

for filename in os.listdir(folder_path):
    filepath = os.path.join(folder_path, filename)
    image = cv2.imread(filepath)
    if (image is not None):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        gray = cv2.fastNlMeansDenoising(gray, 10, 15, 7, 21)
        blurred = cv2.GaussianBlur(gray, (15, 15), 2)
        canny = cv2.Canny(blurred, 20, 40)
        kernel = np.ones((5, 5), np.uint8)


        dilated = cv2.dilate(canny, kernel, iterations=25)
        thresh = cv2.threshold(dilated, 0, 255, cv2.THRESH_BINARY)[1]
        contours = cv2.findContours(dilated.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

        # Create a copy of the original image to draw rectangles
        image_copy = image.copy()

        contour_area = [] 

        for cnt in contours:
            epsilon = 0.1 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            (x, y, w, h) = cv2.boundingRect(approx)
            if (len(approx) == 4):
                contour_area.append(w * h)

        average_area = sum(contour_area) / len(contour_area)
        for cnt in contours:
            epsilon = 0.1 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            (x, y, w, h) = cv2.boundingRect(approx)
            if (len(approx) == 4):
                if (w * h < average_area/2):
                    cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 10)

        # Stack the original image and the dilated image horizontally
        combined_image = np.hstack((cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR), image_copy))

        # Display the combined image
        cv2.imshow(f"Dilated {25} Iterations", combined_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
