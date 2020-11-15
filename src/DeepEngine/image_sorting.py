import glob
import numpy as np
import os

#function to ouput image blurriness
def blur_score(image):
    array = np.asarray(image, dtype=np.int32)
    dx = np.diff(array)[1:,:]
    dy = np.diff(array, axis=0)[:,1:]
    dnorm = np.sqrt(dx**2 + dy**2)
    sharpness = np.average(dnorm)

    return sharpness

#sharpness threshold
threshold = 1.5

#checking if directory exist
if os.path.isdir('./images') is True:
    #glob images from directory
    files = glob.glob("images/*.jpg")
    if os.path.isdir('./images/trash') is False:
        #create directory to move unsuitable images
        os.mkdir("images/trash")
    for i,file in enumerate(files):
        try:
            image = cv2.imread(file)
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            score = blur_score(img)
            if score < threshold:
                os.rename(file,"images/trash/{0}".format(file[7:]))
        except:
            pass
