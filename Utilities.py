import numpy as np
            
def ConvertToGaryscale(ColoredImage):
    
    Red, Green, Blue = ColoredImage[:,:,0], ColoredImage[:,:,1], ColoredImage[:,:,2]
    GrayscaleImage = 0.299 * Red + 0.587 * Green + 0.114 * Blue

    return GrayscaleImage

def Convolve(Image,Gx,Gy=np.zeros((3,3))):
    if len(Image.shape) == 3:
        Image = ConvertToGaryscale(Image)
    # print(len(Gx[0]))
    # print(len(Gy[0]))

    ImageNumberOfRows,ImageNumberOfColumns=Image.shape
    
    KernalNumberOfRows, KernalNumberOfColumns = Gx.shape
    PaddedHight = int((KernalNumberOfRows - 1) / 2)
    PaddedWidth = int((KernalNumberOfColumns - 1) / 2)
    PaddedImage = np.zeros((ImageNumberOfRows + (2 * PaddedHight), ImageNumberOfColumns + (2 * PaddedWidth)))
    PaddedImage[ PaddedHight : PaddedImage.shape[0] - PaddedHight,PaddedWidth : PaddedImage.shape[1] - PaddedWidth,] = Image
    ResultantImage = np.ones([ImageNumberOfRows,ImageNumberOfColumns]) 

    for row in range(ImageNumberOfRows):
        for column in range(ImageNumberOfColumns):

            if(Gy.any() == 0):
                ResultantImage[row,column]=np.sum(np.multiply(Gx,PaddedImage[row:row+KernalNumberOfRows,column:column+KernalNumberOfColumns]))  
            else: 
                PixelValueX=np.sum(np.multiply(Gx,PaddedImage[row:row+KernalNumberOfRows,column:column+KernalNumberOfColumns])) 
                PixelValueY=np.sum(np.multiply(Gy,PaddedImage[row:row+KernalNumberOfRows,column:column+KernalNumberOfColumns]))
                ResultantImage[row,column]=np.sqrt(PixelValueX**2+PixelValueY**2)
              
    return ResultantImage
    
def Threshold(image,High,Low , Weak):
    ResultantImage = np.zeros(image.shape)
    HighNumbersRow, HighNumbersColumn = np.where(image >= High)
    LowNumbersRow, LowNumbersColumn = np.where((image <= High) & (image >= Low))
    ResultantImage[HighNumbersRow, HighNumbersColumn] = 255
    ResultantImage[LowNumbersRow, LowNumbersColumn] =Weak 
    return ResultantImage  

def GaussianFilter(Image,Sigma):
    FilterSize = 2 * int(4 * Sigma + 0.5) + 1
    GAUSSIAN_KERNEL = np.zeros((FilterSize, FilterSize))
    FirstRow= FilterSize//2
    FirstColumn= FilterSize//2    
    for i in range(-FirstRow, FirstRow+1):
        for j in range(-FirstColumn, FirstColumn+1):
            x1 = 2*np.pi*(Sigma**2)
            x2 = np.exp(-(i**2 + j**2)/(2* Sigma**2))
            GAUSSIAN_KERNEL[i+FirstRow, j+FirstColumn] = x2/x1    
    ResultantImage = Convolve(Image, GAUSSIAN_KERNEL,np.zeros(GAUSSIAN_KERNEL.shape))
    return ResultantImage

def Gradient(image):
    SobelKernalX=np.array([[-1,0,1],
                           [-2,0,2],
                           [-1,0,1]])                                                 
    SobelKernalY=np.flip(SobelKernalX.transpose())
    SobelVertical=Convolve(image,SobelKernalX)
    SobelHorizontal=Convolve(image,SobelKernalY)   
    GradientTheta = np.arctan2(SobelHorizontal,SobelVertical)
    GradientTheta=np.rad2deg(GradientTheta) # from radian to degree
    GradientTheta += 180
    GradientMagnitude = np.sqrt(SobelHorizontal**2+SobelVertical**2)
    GradientMagnitude *= 255.0 / GradientMagnitude.max()   
    return GradientMagnitude , GradientTheta 

def NonMaxSuppression(Magnitude,Theta):
    Rows , Columns = Magnitude.shape
    ResultantImage = np.zeros(Magnitude.shape)
    PI = 180
    for row in range(Rows-2):
        for column in range(Columns-2):
            Direction = Theta[row, column]
            # print(Direction)
            if (0 <= Direction < PI / 8) or (15 * PI / 8 <= Direction <= 2 * PI):
                BeforPixel = Magnitude[row, column - 1]
                AfterPixel = Magnitude[row, column + 1]
 
            elif (PI / 8 <= Direction < 3 * PI / 8) or (9 * PI / 8 <= Direction < 11 * PI / 8):
                BeforPixel = Magnitude[row + 1, column - 1]
                AfterPixel = Magnitude[row - 1, column + 1]
 
            elif (3 * PI / 8 <= Direction < 5 * PI / 8) or (11 * PI / 8 <= Direction < 13 * PI / 8):
                BeforPixel = Magnitude[row - 1, column]
                AfterPixel = Magnitude[row + 1, column]
 
            else:
                BeforPixel = Magnitude[row - 1, column - 1]
                AfterPixel = Magnitude[row + 1, column + 1]
 
            if Magnitude[row, column] >= BeforPixel and Magnitude[row, column] >= AfterPixel:
                ResultantImage[row, column] = int(Magnitude[row, column])
    return ResultantImage


def Hysteresis(image, Weak):
    Rows, Columns = image.shape
    TopToBottom = image.copy()
    for row in range(1,Rows-2):
        for col in range(1,Columns-2):
            if TopToBottom[row, col] == Weak:
                if TopToBottom[row, col + 1] == 255 or TopToBottom[row, col - 1] == 255 or TopToBottom[row - 1, col] == 255 or TopToBottom[
                    row + 1, col] == 255 or TopToBottom[
                    row - 1, col - 1] == 255 or TopToBottom[row + 1, col - 1] == 255 or TopToBottom[row - 1, col + 1] == 255 or TopToBottom[
                    row + 1, col + 1] == 255:
                    TopToBottom[row, col] = 255
                else:
                    TopToBottom[row, col] = 0

    BottomToTop = image.copy() 
    for row in range(Rows - 2, 1, -1):
        for col in range(Columns - 2, 1, -1):
            if BottomToTop[row, col] == Weak:
                if BottomToTop[row, col + 1] == 255 or BottomToTop[row, col - 1] == 255 or BottomToTop[row - 1, col] == 255 or BottomToTop[
                    row + 1, col] == 255 or BottomToTop[
                    row - 1, col - 1] == 255 or BottomToTop[row + 1, col - 1] == 255 or BottomToTop[row - 1, col + 1] == 255 or BottomToTop[
                    row + 1, col + 1] == 255:
                    BottomToTop[row, col] = 255
                else:
                    BottomToTop[row, col] = 0

    RightToLeft = image.copy()
    for row in range(1, Rows-2):
        for col in range(Columns - 2, 1, -1):
            if RightToLeft[row, col] == Weak:
                if RightToLeft[row, col + 1] == 255 or RightToLeft[row, col - 1] == 255 or RightToLeft[row - 1, col] == 255 or RightToLeft[
                    row + 1, col] == 255 or RightToLeft[
                    row - 1, col - 1] == 255 or RightToLeft[row + 1, col - 1] == 255 or RightToLeft[row - 1, col + 1] == 255 or RightToLeft[
                    row + 1, col + 1] == 255:
                    RightToLeft[row, col] = 255
                else:
                    RightToLeft[row, col] = 0

    LeftToRight = image.copy() 
    for row in range(Rows - 2, 1, -1):
        for col in range(1, Columns-2):
            if LeftToRight[row, col] == Weak:
                if LeftToRight[row, col + 1] == 255 or LeftToRight[row, col - 1] == 255 or LeftToRight[row - 1, col] == 255 or LeftToRight[
                    row + 1, col] == 255 or LeftToRight[
                    row - 1, col - 1] == 255 or LeftToRight[row + 1, col - 1] == 255 or LeftToRight[row - 1, col + 1] == 255 or LeftToRight[
                    row + 1, col + 1] == 255:
                    LeftToRight[row, col] = 255
                else:
                    LeftToRight[row, col] = 0
 
    ResultantImage = TopToBottom + BottomToTop + RightToLeft + LeftToRight
    ResultantImage[ResultantImage > 255] = 255 
    return ResultantImage