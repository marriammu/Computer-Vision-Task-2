import cv2
import snake
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
# Process command line arguments
file_to_load = "example.jpg"
if len(sys.argv) > 1:
    file_to_load = sys.argv[1]

# Loads the desired image
image = plt.imread(file_to_load)

# Creates the snak
snake = snake.Snake( image, closed = True )

# Window, window name and trackbars
snake_window_name = "Snakes"
controls_window_name = "Controls"
cv2.namedWindow( snake_window_name )
cv2.namedWindow( controls_window_name )
cv2.createTrackbar( "Alpha", controls_window_name, math.floor( snake.alpha * 100 ), 100, snake.set_alpha )
cv2.createTrackbar( "Beta",  controls_window_name, math.floor( snake.beta * 100 ), 100, snake.set_beta )
cv2.createTrackbar( "Gamma", controls_window_name, math.floor( snake.gamma * 100 ), 100, snake.set_gamma )

while(True):
    snakeImg = snake.visualize()
    x = []
    y = []
    for i in range(len(snake.points)):
        x.append(snake.points[i][0])
        y.append(snake.points[i][1])
    #calculate area
    area=0.5*np.sum(y[:-1]*np.diff(x) - x[:-1]*np.diff(y))
    area=np.abs(area)
    #calculate preimeter
    perimeter = snake.get_length()
    print(area)
    print("............")
    print(perimeter)
    cv2.imshow( snake_window_name, snakeImg )
    snake_changed = snake.step()
    
#Stops looping when ESC pressed
    k = cv2.waitKey(33)
    if k == 27:
        break

cv2.destroyAllWindows()

