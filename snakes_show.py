import matplotlib.pyplot as plt
import snake
import numpy as np
def snakes(image,x,y,z):
    img = snake.Snake( image, closed = True )
    img.set_alpha(x)
    img.set_beta(y)
    img.set_gamma(z)
    for i in range(500):
        snakeImg = img.visualize()
        x = []
        y = []
        for i in range(len(img.points)):
            x.append(img.points[i][0])
            y.append(img.points[i][1])
        area=0.5*np.sum(y[:-1]*np.diff(x) - x[:-1]*np.diff(y))
        area=np.abs(area)
        perimeter = img.get_length()
        snake_changed = img.step() 
    plt.imshow(snakeImg)
    plt.show()
    print("area=",area)
    print("............")
    print("perimeter=",perimeter)