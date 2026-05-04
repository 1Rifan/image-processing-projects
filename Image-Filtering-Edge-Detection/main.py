# imprort necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# read and convert image to grayscale
image_rbg = cv2.imread('C:\Image-Filtering-Edge-Detection\images\sample3.jpg')
image = cv2.cvtColor(image_rbg, cv2.COLOR_BGR2GRAY)

#Part 1: A. Image Filtering in the Spatial Domain

# Filter function 
def filter(img, kernel):
    # Get kernel height and width
    kernel_height, kernel_width  = kernel.shape
    
    # Adding zero padding around the image
    padded_image  = np.pad(img, ((1, 1), (1, 1)), mode='constant')

    # Initialize output image with the same shape as input
    filtered_image  = np.zeros_like(img)

    # Loop over every pixel in the original image
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # The region of interest (same size as kernel)
            region = padded_image[i:i+kernel_height, j:j+kernel_width]
            # Apply the filter
            filtered_image[i, j] = np.sum(region * kernel)

    return filtered_image 

# Mean filter
Mean_kernel = 1/9 * np.ones((3,3))  

# Gaussian filter
Gaussian_kernel = 1/16 * np.array([
    [1,2,1],
    [2,4,2],
    [1,2,1]
])

# filter function calls
mean_result = filter(image, Mean_kernel)
gaussian_result = filter(image, Gaussian_kernel)

# result
# Original vs Mean Filter
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(image, cmap='gray')

plt.subplot(1,2,2)
plt.title("Mean Filter")
plt.imshow(mean_result, cmap='gray')

plt.show()

# Original vs Gaussian Filter
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(image, cmap='gray')

plt.subplot(1,2,2)
plt.title("Gaussian Filter")
plt.imshow(gaussian_result, cmap='gray')

plt.show()

#Part 1: B. Image Filtering in the Frequency Domain

# Fourier Transform
f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)

rows, cols = image.shape
crow, ccol = rows//2 , cols//2

# Low-pass filter mask
mask = np.zeros((rows, cols), np.uint8)
r = 50
center = (crow, ccol)

for i in range(rows):
    for j in range(cols):
        if (i-center[0])**2 + (j-center[1])**2 <= r*r:
            mask[i,j] = 1

# applying filter
filtered = fshift * mask

# Inverse FFT
f_ishift = np.fft.ifftshift(filtered)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

# result
# Original vs Low-pass Filter
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(image, cmap='gray')

plt.subplot(1,2,2)
plt.title("Low-pass Result")
plt.imshow(img_back, cmap='gray')

plt.show()


#Part 2: Implementation and Analysis of Edge Detection Techniques
# read and convert image to grayscale
image_rbg2 = cv2.imread('C:\image-filtering-edge-detection\images\sample1.jpg')
image2 = cv2.cvtColor(image_rbg2, cv2.COLOR_BGR2GRAY)

image_rbg3 = cv2.imread('C:\image-filtering-edge-detection\images\sample2.jpg')
image3 = cv2.cvtColor(image_rbg3, cv2.COLOR_BGR2GRAY)

# Sobel
sobel_x = cv2.Sobel(image2, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image2, cv2.CV_64F, 0, 1, ksize=3)
sobel = np.sqrt(sobel_x**2 + sobel_y**2)

sobel_x2 = cv2.Sobel(image3, cv2.CV_64F, 1, 0, ksize=3)
sobel_y2 = cv2.Sobel(image3, cv2.CV_64F, 0, 1, ksize=3)
sobel2 = np.sqrt(sobel_x2**2 + sobel_y2**2)

# LoG
log = cv2.GaussianBlur(image2, (3,3), 0)
log = cv2.Laplacian(log, cv2.CV_64F)

log2 = cv2.GaussianBlur(image3, (3,3), 0)
log2 = cv2.Laplacian(log2, cv2.CV_64F)

# Canny
canny = cv2.Canny(image2, 100, 200)

canny2 = cv2.Canny(image3, 100, 200)

# result 
# 1st image:
# Original vs Sobel
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(image2, cmap='gray')
plt.subplot(1,2,2)
plt.title("Sobel")
plt.imshow(sobel, cmap='gray')
plt.show()

# Original vs LoG
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(image2, cmap='gray')
plt.subplot(1,2,2)
plt.title("LoG")
plt.imshow(log, cmap='gray')
plt.show()

# Original vs Canny
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(image2, cmap='gray')
plt.subplot(1,2,2)
plt.title("Canny")
plt.imshow(canny, cmap='gray')
plt.show()

# 2nd image:
# Original vs Sobel
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(image3, cmap='gray')
plt.subplot(1,2,2)
plt.title("Sobel")
plt.imshow(sobel2, cmap='gray')
plt.show()

# Original vs LoG
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(image3, cmap='gray')
plt.subplot(1,2,2)
plt.title("LoG")
plt.imshow(log2, cmap='gray')
plt.show()

# Original vs Canny
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(image3, cmap='gray')
plt.subplot(1,2,2)
plt.title("Canny")
plt.imshow(canny2, cmap='gray')
plt.show()
