import glob
import cv2
import numpy as np
import shutil
import os


class skinDetector(object):
    """
    Author: Jean Vitor de Paulo
    Altered: Shozen Dan
    Date: 3/12/2020
    """
    # Class constructor
    def __init__(self, img):

        #self.image = cv2.resize(self.image,(600,600),cv2.INTER_AREA)
        self.image = img
        self.HSV_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.YCbCr_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2YCR_CB)
        self.binary_mask_image = self.HSV_image
        self.output = None

    # Function to process the image and segment the skin using the HSV and YCbCr colorspaces, followed by the Watershed algorithm
    def find_skin(self):
        self.__color_segmentation()
        self.__region_based_segmentation()

        # Apply a threshold to an HSV and YCbCr images, the used values were based on current research papers along with some
    # empirical tests and visual evaluation
    def __color_segmentation(self):
        lower_HSV_values = np.array([0, 40, 0], dtype="uint8")
        upper_HSV_values = np.array([25, 255, 255], dtype="uint8")
        lower_YCbCr_values = np.array((0, 138, 67), dtype="uint8")
        upper_YCbCr_values = np.array((255, 173, 133), dtype="uint8")
        # A binary mask is returned. White pixels (255) represent pixels that fall into the upper/lower.
        mask_YCbCr = cv2.inRange(
            self.YCbCr_image, lower_YCbCr_values, upper_YCbCr_values)
        mask_HSV = cv2.inRange(
            self.HSV_image, lower_HSV_values, upper_HSV_values)
        self.binary_mask_image = cv2.add(mask_HSV, mask_YCbCr)
        
    # Function that applies Watershed and morphological operations on the thresholded image
    def __region_based_segmentation(self):
        # morphological operations
        image_foreground = cv2.erode(self.binary_mask_image, None, iterations = 3)     	#remove noise
        dilated_binary_image = cv2.dilate(self.binary_mask_image, None, iterations = 3)   #The background region is reduced a little because of the dilate operation
        ret, image_background = cv2.threshold(dilated_binary_image, 1,128,cv2.THRESH_BINARY)  #set all background regions to 128
        image_marker = cv2.add(image_foreground, image_background)  # add both foreground and backgroud, forming markers. The markers are "seeds" of the future image regions.
        image_marker32 = np.int32(image_marker)  # convert to 32SC1 format
        cv2.watershed(self.image, image_marker32)
        m = cv2.convertScaleAbs(image_marker32)  # convert back to uint8
        # bitwise of the mask with the input image
        ret, image_mask = cv2.threshold(m, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.output = cv2.bitwise_and(self.image,self.image,mask = image_mask)

# finds the area of skin, and crops it into square image
# higher the dp_value, more sensitive the circle detection


class circleCropper:
    def __init__(self, size=500, dp = [3, 6], bt = 1):
        self.size = size  # target image size
        self.dp = dp
        self.bt = bt

    def denoise(self, img):
        """Removes noise using by applying Gaussian blur 3 times"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flat = cv2.GaussianBlur(gray, (3, 3), 0)
        flat = cv2.GaussianBlur(flat, (5, 5), 0)
        flat = cv2.GaussianBlur(flat, (7, 7), 0)

        return flat

    def detect(self, img, dp):
        """Detect circle and crops accordingly"""
        X, Y = img.shape
        circles = cv2.HoughCircles(
            img,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            param1=70,
            param2=50,
            minDist=2000,
            minRadius=300,
            maxRadius=int(X * np.pi)
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                r_adj = r * 1.5 * np.sin(np.pi/8)
                x1, x2, y1, y2 = int(x - r_adj), int(x + r_adj), int(y - r_adj), int(y + r_adj)
                return (x1, x2, y1, y2)

        return None  # No circle detected or image is too blurred

    def crop(self, input_dir, output_dir):
        """Main method"""
        # Sanity check
        if input_dir == None:
            raise Exception('Please specify input directory')
        if output_dir == None:
            raise Exception('Please specify output directory')

        # Check if input_dir exists
        if not os.path.exists(input_dir):
            raise Exception('Input directory cannot be found')

        # Overwrite output_dir if it exists
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        # Create output_dir if it doesn't exist
        if not os.path.exists(output_dir):
            print('Creating output directory')
            os.mkdir(output_dir)
            os.mkdir(os.path.join(output_dir, 'success'))
            os.mkdir(os.path.join(output_dir, 'failure'))

        files = glob.glob(os.path.join(input_dir, "*"))
        skip = ''
        corrupt = list()

        for i, f in enumerate(files):
            # Process display
            fname = os.path.basename(f)
            l1 = "progress : {0} / {1} ({2}%)".format(i+1,
                                                      len(files), int((i+1)/len(files)*100))
            l2 = "skipped : {}".format(os.path.basename(skip))
            multi_line = l1 + "\n" + l2
            ret_depth = '\033[F' * multi_line.count('\n')
            print('{}{}'.format(ret_depth, multi_line), end='', flush=True)

            try:
                img = cv2.imread(f)
                copy = img.copy()
                # Remove noise
                skd = skinDetector(img)
                skd.find_skin()
                flat = self.denoise(skd.output)
                # Main process
                crop_coords = self.detect(flat, self.dp[0])
                if crop_coords is not None:
                    x1, x2, y1, y2 = crop_coords
                    cropped = copy[y1:y2, x1:x2, :]
                    cv2.imwrite(os.path.join(
                        output_dir, 'success', fname), cropped)
                else:
                    crop_coords = self.detect(flat, self.dp[1])
                    if crop_coords is not None:
                        x1, x2, y1, y2 = crop_coords
                        cropped = copy[y1:y2, x1:x2, :]
                        cv2.imwrite(os.path.join(
                            output_dir, 'success', fname), cropped)
                    else:
                        cv2.imwrite(os.path.join(
                            output_dir, 'failure', fname), img)

            except:
                fname = os.path.basename(f)
                skip = fname
                corrupt.append(fname)
                pass

        print("\nList of corrupted File(s)")
        for item in corrupt:
            print(item)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Crop images')
    parser.add_argument('-i', '--i', dest='input_dir',
                        help='input directory path')
    parser.add_argument('-o', '--o', dest='output_dir',
                        help='output directory path')
    args = parser.parse_args()

    cropper = circleCropper()
    cropper.crop(args.input_dir, args.output_dir)
