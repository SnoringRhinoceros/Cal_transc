import cv2

# Load the image
image = cv2.imread('data/test/test_cal.png')

# Apply Canny edge detection
edges = cv2.Canny(image, 100, 200)

# Save the result
cv2.imwrite('data/test_cal_edges.png', edges)
