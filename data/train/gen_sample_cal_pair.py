from PIL import Image, ImageDraw

# Auto generates the coordinates of the edges of the calendar

# Each calendar rect is 116.5(width)x80.5(length) pixels
# The top left pixel is (2,66) and the bottom right is (816,470)

# Dimensions of each rectangle
rect_width = 116.5
rect_length = 80.5

# Starting point (top-left corner) of the grid
start_x = 2
start_y = 66

# Ending point (bottom-right corner) of the grid
end_x = 816
end_y = 470

# Calculate the number of rows and columns
num_cols = int((end_x - start_x) / rect_width) + 1
num_rows = int((end_y - start_y) / rect_length) + 1

# Open the original image
original_image = Image.open("./data/train/train_cal.png")
width, height = original_image.size

# Create a new image with a transparent background
overlay = Image.new('RGBA', (width, height), (255, 255, 255, 0))
draw = ImageDraw.Draw(overlay)

# Function to draw the coordinates of each rectangle
def draw_coordinates():
    with open("./data/train/pair.txt", "w") as file:
        for row in range(num_rows):
            for col in range(num_cols):
                top_left = (start_x + col * rect_width, start_y + row * rect_length)
                bottom_right = (top_left[0] + rect_width, top_left[1] + rect_length)
                file.write(f"[{row}, {col}, {top_left}, {bottom_right}]\n")
                # Commented out drawing code
                # draw.rectangle([top_left, bottom_right], outline="red", fill=(255, 0, 0, 128))

# Draw the rectangles
draw_coordinates()

# Composite the overlay with the original image
# combined = Image.alpha_composite(original_image.convert('RGBA'), overlay)

# Save the resulting image
# combined.save("./data/train/train-Calendar-With-Boxes.png")

print("Coordinates saved to pair.txt")
# print("Image saved as train-Calendar-With-Boxes.png")