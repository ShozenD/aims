import glob
import cv2
import numpy as np
import shutil
import os

# finds the area of skin, and crops it into square image
# higher the dp_value, more sensitive the circle detection
class circleCropper:
    def __init__(self, size = 500, dp = [3, 6], bt = 1):
        self.size = size # target image size
        self.dp = dp
        self.bt = bt

    def denoise(self, img):
        """Removes noise using by applying Gaussian blur 3 times"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flat = cv2.GaussianBlur(gray, (3,3), 0)
        flat = cv2.GaussianBlur(flat, (5,5), 0)
        flat = cv2.GaussianBlur(flat, (7,7), 0)

        return flat

    def variance_of_laplacian(self, img):
        return cv2.Laplacian(img, cv2.CV_64F).var()

    """
    def blur_score(self, img):
        percentage = self.size/img.shape[1] # resize percentage
        w, h = int(img.shape[1] * percentage), int(img.shape[0] * percentage)
        resize = cv2.resize(img, (w, h))

        # calculating average gradient magnitude(sharpness)
        # arr = np.array(resize, dtype=np.int32)
        dx, dy = np.diff(resize)[1:,:], np.diff(resize, axis=0)[:,1:] 
        dnorm = np.sqrt(dx**2 + dy**2)
        sharpness = np.average(dnorm)

        return sharpness
    """
    
    def detect(self, img, dp):
        """Detect circle and crops accordingly"""
        X, Y = img.shape
        circles = cv2.HoughCircles(
            img, 
            cv2.HOUGH_GRADIENT,
            dp = dp,
            param1 = 70,
            param2 = 50,
            minDist = 2000,
            minRadius = 300,
            maxRadius = int(X * np.pi)
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                r_adj = r * np.sin(np.pi/8)
                x1, x2, y1, y2 = int(x - r_adj), int(x + r_adj), int(y - r_adj), int(y + r_adj)
                cropped = img[y1:y2, x1:x2]
                if self.variance_of_laplacian(cropped) > self.bt: 
                    return (x1, x2, y1, y2)

        return None # No circle detected or image is too blurred

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

        files = glob.glob(os.path.join(input_dir,"*"))
        skip = ''
        corrupt = list()

        for i, f in enumerate(files):
            # Process display
            fname = os.path.basename(f)
            l1 = "progress : {0} / {1} ({2}%)".format(i+1, len(files), int((i+1)/len(files)*100))
            l2 = "skipped : {}".format(os.path.basename(skip))
            multi_line = l1 + "\n" + l2
            ret_depth = '\033[F' * multi_line.count('\n')
            print('{}{}'.format(ret_depth, multi_line), end='', flush = True)

            try:
                img = cv2.imread(f)
                copy = img.copy()
                # Remove noise
                flat = self.denoise(img)
                # Main process
                crop_coords = self.detect(flat, self.dp[0])
                if crop_coords is not None:
                    x1, x2, y1, y2 = crop_coords
                    cropped = copy[y1:y2,x1:x2,:]
                    cv2.imwrite(os.path.join(output_dir, 'success', fname), cropped)
                else:
                    crop_coords = self.detect(flat, self.dp[1])
                    if crop_coords is not None:
                        x1, x2, y1, y2 = crop_coords
                        cropped = copy[y1:y2,x1:x2,:]
                        cv2.imwrite(os.path.join(output_dir, 'success', fname), cropped)
                    else:
                        cv2.imwrite(os.path.join(output_dir, 'failure', fname), img)

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
    parser.add_argument('-i', '--i', dest='input_dir', help='input directory path')
    parser.add_argument('-o', '--o', dest='output_dir', help='output directory path')
    args = parser.parse_args()

    cropper = circleCropper()
    cropper.crop(args.input_dir, args.output_dir)