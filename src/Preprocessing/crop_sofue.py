import glob
import cv2
import numpy as np
import shutil
import os

try:
    if os.path.exists("./new") is True:
        shutil.rmtree('./new')
    os.mkdir("./new") #cropped image goes here (dp=3)
    os.mkdir("./new/retry") #cropped image with higher dp goes here (dp=6)
    os.mkdir("./new/fail") #irrelevant and failed image goes here
except:
    pass
#calculates the blurriness of the image
#note: changing target_size will influence sharpness score
def blur_score(image):
    target_size = 500 # the target size of image
    percentage = target_size/image.shape[1] #resize percentage
    w, h = int(image.shape[1] * percentage),int(image.shape[0] * percentage)
    dsize = (w, h)
    resize = cv2.resize(image, dsize)
    gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
    #calculating average gradient magnitude(sharpness)
    array = np.asarray(gray, dtype=np.int32)
    dx = np.diff(array)[1:,:]
    dy = np.diff(array, axis=0)[:,1:]
    dnorm = np.sqrt(dx**2 + dy**2)
    sharpness = np.average(dnorm)

    return sharpness
#finds the area of skin, and crops it into square image
#higher the dp_value, more sensitive the circle detection
def detection(image,dp_value,save_location):
    try:
        circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT,dp=dp_value,param1=70,param2=50,minDist=2000,minRadius=300,maxRadius=600)
        if circles is not None:
           circles = np.round(circles[0, :]).astype("int")
           for (x, y, r) in circles:
              cropped = output[int(y-(r/2)):int(y+(r/2)),int(x-(r/2)):int(x+(r/2))]
              h,w,c = cropped.shape
              if h == w:
                  score = blur_score(cropped)
                  if score > 0.5: #0.5 is the sharpness threshold (can be changed)
                      cv2.imwrite(save_location, cropped)
                  else:
                      cv2.imwrite("./new/fail/{0}.jpg".format(i), cropped)
              return True
        else:
            return False
    except:
        return False
#glob all image files
files = glob.glob("images/*.png")
skip = 0
corrupt = []
print("\n")
for i,file in enumerate(files):
    #process display
    l1="progress : {0} / {1} ({2}%)".format(i+1,len(files),int((i+1)/len(files)*100))
    l2="skipped : {0}".format(skip)
    multi_line = l1+"\n"+l2
    ret_depth = '\033[F' * multi_line.count('\n')
    print('{}{}'.format(ret_depth, multi_line), end='', flush = True)
    try:
        image = cv2.imread(file)
        output = image.copy()
        #remove noise
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flat=cv2.GaussianBlur(img,(3,3),0)
        flat=cv2.GaussianBlur(flat,(5,5),0)
        flat=cv2.GaussianBlur(flat,(7,7),0)
        #main process
        result=detection(flat,3,"./new/{0}.jpg".format(i))
        if result is False:
            result=detection(flat,6,"./new/retry/{0}.jpg".format(i))
            if result is False:
                cv2.imwrite("./new/fail/{0}.jpg".format(i), output)
    except:
        skip+=1
        corrupt.append(file)
        pass
print("\nList of Skipped File(s)")
for item in corrupt:
    print(item)
