import numpy as np
import cv2
def iris_enhancement(img):
  # convert input to grayscale image 
  #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) commented out as this is not needed providing the input is in grayscale
  # get the number of rows and columns from the image
  rows, columns = img.shape
  # set the block_size variable to be the length/width in pixels for each block constituent of the image
  # the paper uses block size 16
  block_size = 16
  # divide both the rows and columns by the block_size variable so the function knows how many multiples of block_size to go through
  num_rows = rows // block_size
  num_cols = columns // block_size
  # instantiate a list for the block constituent means
  block_means = []
  # using a double for loop get the (i,j) block 
  for i in range(num_rows):
      for j in range(num_cols):
          # set the block array variable by subsetting the original 
          block = img[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
          # create the mean variable 
          mean_value = block.mean()
          # append the mean to the block_means list
          block_means.append(mean_value)
          
  # create an empty array variable called bicubic_interpolated_means to store the block means bicubically interpolated 
  # to the size of the original image shape
  
  bicubic_interpolated_means = np.empty((rows, columns), dtype=np.float32)
  
  # reshape the block means to a 2D array with the shape of the number of blocks in both rows and columns
  block_means_array = np.array(block_means).reshape(num_rows, num_cols)
  
  # perform bicubic interpolation on the block means using a double for loop
  for i in range(num_rows):
      for j in range(num_cols):
          # set the (i,j)th block (meaning a block of block_size X block_size pixels) of the bicubic_interpolated_means variable 
          # to equal the mean of that block
          bicubic_interpolated_means[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = block_means_array[i, j]
  
  # convert the bicubic interpolated means to the original image data type
  bicubic_interpolated_means = bicubic_interpolated_means.astype(np.uint8)
  # use the cv2.merge() method to create a "background noise image"
  background_image = cv2.merge([bicubic_interpolated_means] * 1)
  # subtract the background_image from the original grayscale image and store this array as "partially_enhanced"
  partially_enhanced = (img-background_image)
  
  # make a copy of the partially_enhanced array to become the histogram equalized image
  enhanced_image = np.copy(partially_enhanced)
  # set the height and width variables using the shape method
  height, width = enhanced_image.shape
  # make sure the enhanced_image data type is the correct 8-bit depth for the equalizeHist function to work
  if enhanced_image.dtype != np.uint8:
    enhanced_image = cv2.convertScaleAbs(enhanced_image)
  # create a new variable for region size called "region_size" which is 2*block_size as the Li paper uses 32x32 regions (block size is 16 in the paper)
  region_size = block_size*2
  # instantiate a double for loop that goes through each region_size by region_size block of the image
  # and converts the image's region to its hist equalized version
  for y in range(0, height, region_size):
      for x in range(0, width, region_size):
          # set the region variable to be the subset of the partially enhanced image array within the region
          region = enhanced_image[y:y+region_size, x:x+region_size]
          # use opencv2.equalizeHist to get the equalized version of the region and store it as "equalized_region"
          equalized_region = cv2.equalizeHist(region)
          # set the enhanced_image array to have the equalized_region values for the region the loop is on
          enhanced_image[y:y+region_size, x:x+region_size] = equalized_region
  return enhanced_image
