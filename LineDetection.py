from utils import *

def Threshold(image,High,Low , Weak):
    ResultantImage = np.zeros(image.shape)
    HighNumbersRow, HighNumbersColumn = np.where(image >= High)
    LowNumbersRow, LowNumbersColumn = np.where((image <= High) & (image >= Low))
    ResultantImage[HighNumbersRow, HighNumbersColumn] = 255
    ResultantImage[LowNumbersRow, LowNumbersColumn] =Weak 
    return ResultantImage  

def Canny(image):
    Weak=50
    GaussianImage=GaussianFilter(image,1)
    Magnitude , Theta = Gradient(GaussianImage)
    Suppression=NonMaxSuppression(Magnitude,Theta)
    Thresholding=Threshold(Suppression,50,5,Weak)
    FinalImage=Hysteresis(Thresholding,Weak)  
    return FinalImage