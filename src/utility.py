'''Module containing utility functions for the project.'''
# pylint: disable=maybe-no-member

import os
import cv2
import numpy as np
from sklearn.cluster import KMeans



def save_mask_as_image(mask, output_path):
    """
    Saves the logical mask as an image at the given path.

    :param mask: Logical mask (numpy array)
    :param output_path: Path to the output file
    """
    mask = mask.astype(np.uint8)
    cv2.imwrite(output_path, mask)
    print(f"Saved mask to {output_path}")

def get_click_coordinates(image, num_points = 2):
    """
    Displays an image and allows the user to click on it to select coordinates.
    Args:
        image (numpy.ndarray): The image on which the user will click to select points.
        num_points (int): The number of points the user will select.
    Returns:
        numpy.ndarray: An array of shape (num_points, 2) containing the coordinates of the points 
                       selected by the user.
    """

    coordinates = []

    def mouse_callback(event, x, y, flags, param): # pylint: disable=unused-argument
        if event == cv2.EVENT_LBUTTONDOWN:
            coordinates.append((x, y))
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Mark the point on the image
            cv2.imshow('Click to select point', image)
            if len(coordinates) == num_points:
                cv2.destroyAllWindows()

    cv2.imshow('Click to select point', image)
    cv2.setMouseCallback('Click to select point', mouse_callback)
    cv2.waitKey(0)

    coordinates = np.array(coordinates).reshape(-1, 2)
    return coordinates

def distribute_points_using_kmeans(mask, num_points):
    """
    Distribute a specified number of points optimally on a binary mask using k-means clustering.

    Args:
        mask (np.ndarray): Binary mask array where points should be distributed.
        num_points (int): Number of points to distribute.

    Returns:
        np.ndarray: Array of [x, y] coordinates for the distributed points.
    """
    # Get valid points from the mask
    y_indices, x_indices = np.where(mask == 1)
    if len(y_indices) == 0:
        return np.array([])

    # Combine coordinates into a single array for clustering
    coordinates = np.column_stack((x_indices, y_indices))

    # Use k-means clustering to find clusters
    kmeans = KMeans(n_clusters=min(num_points, len(coordinates)), random_state=0, n_init='auto')
    kmeans.fit(coordinates)
    cluster_centers = kmeans.cluster_centers_

    # Round cluster centers to nearest integer coordinates
    distributed_points = np.round(cluster_centers).astype(int)

    return distributed_points


class ImagePathUtility:
    """Utility class for handling image paths and saving masks as images."""
    @staticmethod
    def get_image_paths(input_dir: str, image_extensions: tuple = (".jpg", ".jpeg", ".png", ".tiff", ".bmp")) -> list:
        """
        Retrieves image paths from the specified directory given the supported extensions.
        
        :param input_dir: Directory containing images.
        :param image_extensions: List of supported image extensions.
        :return: List of image paths.
        """
        image_paths = [
            os.path.join(input_dir, f) for f in os.listdir(input_dir)
            if f.lower().endswith(image_extensions)
        ]
        if not image_paths:
            raise ValueError(
                f"No images found in {input_dir} with supported extensions: {image_extensions}"
            )
        print(f"Found {len(image_paths)} images in {input_dir}")
        return image_paths

    @staticmethod
    def save_mask_as_image(mask: np.ndarray, output_path: str) -> None:
        """
        Saves the binary mask as an image at the given path.
        
        :param mask: Binary mask (numpy array).
        :param output_path: Path to the output file.
        """
        mask = mask.astype(np.uint8) * 255
        cv2.imwrite(output_path, mask)
        print(f"Saved mask to {output_path}")

class ImageProcessor:
    """Utility class for image processing operations."""
    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """
        Loads an image from the given file path.
        
        :param image_path: Path to the image file.
        :return: Loaded image as a numpy array.
        """
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_rgb

    @staticmethod
    def rescale(image: np.ndarray, rescale_factor: float) -> np.ndarray:
        """
        Rescales an image by the specified factor.
        
        :param image: Input image as a numpy array.
        :param rescale_factor: Factor to rescale the image.
        :return: Rescaled image.
        """
        return cv2.resize(
            image,
            (int(image.shape[1] * rescale_factor),
             int(image.shape[0] * rescale_factor))
        )

    @staticmethod
    def invert_mask(mask: np.ndarray) -> np.ndarray:
        """
        Inverts a binary mask.
        
        :param mask: Input binary mask (0 or 1).
        :return: Inverted binary mask.
        """
        mask = mask * 255
        return cv2.bitwise_not(mask) / 255

    @staticmethod
    def get_biggest_object_from_mask(mask: np.ndarray) -> np.ndarray:
        """
        Extracts the largest object from a binary mask.
        
        :param mask: Input binary mask (0 or 1).
        :return: Binary mask containing only the largest object.
        """
        mask = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_final = np.zeros_like(mask)
        if contours:
            biggest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask_final, [biggest_contour], -1, 1, cv2.FILLED)
        return mask_final

    @staticmethod
    def erode_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Erodes the mask using a specified kernel size.
        
        :param mask: Input binary mask (0 or 1).
        :param kernel_size: Kernel size for erosion.
        :return: Eroded mask.
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.erode(mask, kernel, iterations=1)

    @staticmethod
    def crop_image(image: np.ndarray, box: tuple) -> np.ndarray:
        """
        Crops the image using the specified bounding box.
        
        :param image: Input image as a numpy array.
        :param box: Bounding box coordinates (x, y, w, h).
        :return: Cropped image.
        """
        x, y, w, h = box
        return image[y:y+h, x:x+w]
