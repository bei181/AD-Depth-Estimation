import sys
import pathlib

import cv2
import imutils
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def show_images(images: list, scale: float = None):
  """Shows a list of images numbered 0..n and waits for a keypress to continue.

  Args:
      images (list): The list of images to display
      scale (float, optional): A scale value for the displayed images. Defaults to None.
  """
  for i, image in enumerate(images):
    if scale is None:
      cv2.imshow(f"{i}", image)
    else:
      cv2.imshow(f"{i}", imutils.resize(image, (int) (image.shape[1] * scale), (int) (image.shape[0] * scale)))

  cv2.waitKey()

def scale_image(img: np.ndarray, scale: float):
  """Scales an image by a given scale value equally in vertical and horizontal direction.

  Args:
      img ([type]): The source image
      scale (float): The scale value

  Returns:
      numpy.ndarray: The resized image
  """
  return cv2.resize(img, ((int) (img.shape[1] * scale), (int) (img.shape[0] * scale)))

def crop_image(img: np.ndarray, position: tuple, width: int, height: int):
  """Crops a given image in a rectangle shape starting from the given position with a width and height.

  Args:
      img (numpy.ndarray): The input image
      position (tuple): The x and y position as tuple representing the top-left corner of the cropped rectangle
      width (int): The width of the cropped image
      height (int): The height of the cropped image

  Returns:
      numpy.ndarray: The cropped image
  """
  return img[position[1]:position[1] + height, position[0]:position[0] + width]

def crop_image_centered(img: np.ndarray, width: int, height: int):
  """Crops a given image in a centered position.

  Args:
      img (numpy.ndarray): The input image
      width (int): The width of the cropped image
      height (int): The height of the cropped image

  Returns:
      numpy.ndarray: The cropped image 
  """
  position = ((img.shape[1] // 2) - (width // 2), (img.shape[0] // 2) - (height // 2))
  
  return crop_image(img, position, width, height)

def plot_image_realtime(img: np.ndarray):
  """Plots a given image using a realtime method (plotly library in this case).

  Args:
      img (numpy.ndarray): The input image to be plotted (1 Channel)
  """
  assert len(img.shape) == 2, "Image has to have 1 channels"

  # https://plotly.com/python/3d-surface-plots/
  fig = go.Figure(data=[go.Surface(z=img)])
  fig.show()

def plot_image(img: np.ndarray):
  """Plots a given image using a non-realtime suited method (matplotlib).

  Args:
      img (numpy.ndarray): The input image to be plotted (1 Channel)
  """
  assert len(img.shape) == 2, "Image has to have 1 channels"

  # https://stackoverflow.com/questions/31805560/how-to-create-surface-plot-from-greyscale-image-with-matplotlib
  # This approach is very slow, need to look for a more realtime oriented way of doing things

  # Create x and y coordinate arrays
  xx, yy = np.mgrid[0: img.shape[0], 0:img.shape[1]]

  fig = plt.figure()
  ax = plt.axes(projection="3d")
  ax.plot_surface(xx, yy, img, rstride=1, cstride=1, cmap=plt.cm.gray, linewidth=0)
  ax.set_title("image plot")

  plt.show()

def convert_image_rgb2gray(img: np.ndarray):
  """Converts an RGB image to a grayscale image while discarding values outside the range of 0 and 255 and taking only the values within the valid range.

  Args:
      img (numpy.ndarray): The RGB input image

  Returns:
      numpy.ndarray: The grayscale image with only 1 channel
  """
  assert len(img.shape) > 2, "Image cannot be grayscale"
  assert img.shape[2] == 3, "Image has to have 3 channels"
 
  result = np.zeros(img.shape[:2], np.uint8)

  def clipped_value(values: np.ndarray):
    return np.max(np.minimum(np.maximum(values, 0), 255), axis=2)

  return clipped_value(img)


def normalize_image(img: np.ndarray):
  """Normalize an images values up to 0..255 (Previous max value will be 255 afterwards).

  Args:
      img (numpy.ndarray): The input image

  Returns:
      tuple(numpy.ndarray, int): A tuple containing the input image and the value by which it has been scaled (Previous max value) 
  """
  minValue = img.min()
  maxValue = img.max()
  result = np.float32((img - minValue) * (255 / (maxValue - minValue)))

  return result, minValue, maxValue 