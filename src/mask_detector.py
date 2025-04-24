import sys
import os
import logging
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from src.SAM_wrapper import SamPredictorWrapper
from src.utility import get_click_coordinates, ImageProcessor

# Configure logging
global_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def distribute_points(mask: np.ndarray,
                      num_points: int,
                      n_init: int = 10,
                      random_state: int = 0) -> np.ndarray:
    """
    Distribute points over a binary mask using k-means clustering.
    """
    y_indices, x_indices = np.where(mask == 1)
    if len(y_indices) == 0:
        return np.array([])
    coords = np.column_stack((x_indices, y_indices))
    kmeans = KMeans(n_clusters=min(num_points, len(coords)),
                    random_state=random_state,
                    n_init=n_init)
    kmeans.fit(coords)
    centers = np.round(kmeans.cluster_centers_).astype(int)
    return centers


class MaskDetectorConfig:
    """
    Configuration containing input and output paths.
    """
    def __init__(self):
        # SAM model settings
        self.model_type = "vit_h"
        self.checkpoint_path = self._get_resource_path(r"models/sam_vit_h.pth")
        # Visualization and resizing
        self.is_display = True
        self.downscale_factor = 5
        # Clustering and mask parameters
        self.num_positive_points = 2
        self.num_negative_points = 12
        self.erode_size = 5
        self.kmeans_n_init = 10
        self.kmeans_random_state = 0
        self.box_roi = None
        self.init_points_positive = None
        self.input_paths = []  # list of source image file paths
        self.output_paths = [] # list of corresponding mask save paths

    def _get_resource_path(self, relative_path: str) -> str:
        base = getattr(sys, 'frozen', False) and sys._MEIPASS or os.path.abspath('.')
        return os.path.join(base, relative_path)

class MaskVisualizer:
    """Handles mask overlay visualization."""
    @staticmethod
    def overlay(image: np.ndarray,
                mask: np.ndarray,
                pos: np.ndarray,
                neg: np.ndarray = None,
                alpha: float = 0.5,
                radius: int = 5,
                thickness: int = 2) -> np.ndarray:
        vis = image.copy()
        color_mask = np.zeros_like(image)
        color_mask[mask == 255] = [0, 255, 0]
        vis = cv2.addWeighted(vis, 1, color_mask, alpha, 0)
        if pos is not None:
            for pt in pos:
                cv2.circle(vis, tuple(pt), radius, (0, 255, 0), thickness)
        if neg is not None:
            for pt in neg:
                cv2.circle(vis, tuple(pt), radius, (255, 0, 0), thickness)
        return vis

    @staticmethod
    def display(image: np.ndarray,
                mask: np.ndarray = None,
                pos: np.ndarray = None,
                neg: np.ndarray = None,
                wait_ms: int = 0) -> None:
        if mask is not None:
            mask_disp = (mask.astype(np.uint8) * 255)
            vis = MaskVisualizer.overlay(image, mask_disp, pos, neg)
            cv2.imshow('Mask', cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            cv2.waitKey(wait_ms)
            cv2.destroyAllWindows()


class ImageProcessingState:
    def __init__(self,
                 pos: np.ndarray = None,
                 neg: np.ndarray = None,
                 current: np.ndarray = None,
                 previous: np.ndarray = None):
        self.points_positive = pos
        self.points_negative = neg
        self.mask_current = current
        self.mask_previous = previous


class MaskDetector:
    """Main class coordinating mask detection pipeline."""
    def __init__(self, cfg: MaskDetectorConfig):
        self.cfg = cfg
        self.logger = global_logger
        try:
            self.logger.info('Initializing SAM predictor with model %s', cfg.model_type)
            self._sam = SamPredictorWrapper(cfg.model_type, cfg.checkpoint_path)
        except Exception as e:
            self.logger.error('Failed to load SAM model: %s', e)
            raise

    def process_images(self) -> None:
        # Iterate explicit input/output pairs
        for src, dst in tqdm(zip(self.cfg.input_paths, self.cfg.output_paths or []),
                     desc='Processing', total=len(self.cfg.input_paths)):
            try:
                img = ImageProcessor.load_image(src)
                img = ImageProcessor.rescale(img, 1/self.cfg.downscale_factor)
                if self.cfg.box_roi is not None:
                    img_proc = ImageProcessor.crop_image(img, self.cfg.box_roi)
                else:
                    img_proc = img

                # Initial selection only for first image
                if not hasattr(self, '_state'):
                    if self.cfg.init_points_positive is None:
                        self.logger.info('Select %d positive points', self.cfg.num_positive_points)
                        pts = get_click_coordinates(cv2.cvtColor(img_proc, cv2.COLOR_RGB2BGR))
                    else:
                        pts = self.cfg.init_points_positive
                    self._state = ImageProcessingState(pos=pts)

                self._state = self.process_single_image(img_proc, self._state)

                # Prepare final mask
                if self.cfg.box_roi:
                    full = np.zeros(img.shape[:2], dtype=np.uint8)
                    x, y, w, h = self.cfg.box_roi
                    full[y:y+h, x:x+w] = self._state.mask_current
                    out_mask = full
                else:
                    out_mask = self._state.mask_current

                os.makedirs(os.path.dirname(dst), exist_ok=True)
                ImageProcessor.save_mask_as_image(out_mask, dst)
            except Exception as e:
                self.logger.error('Error in processing %s: %s', src, e)
                continue

    def process_single_image(self,
                             img: np.ndarray,
                             state: ImageProcessingState) -> ImageProcessingState:
        try:
            masks = self._sam.predict_mask(
                img,
                state.points_positive,
                state.points_negative,
                mask_input=state.mask_current
            )
        except Exception as e:
            self.logger.error('Prediction error: %s', e)
            return state

        state.mask_current = ImageProcessor.get_biggest_object_from_mask(masks[1])
        if self.cfg.is_display:
            MaskVisualizer.display(img, state.mask_current,
                                    state.points_positive,
                                    state.points_negative)

        # Compute next points
        eroded = ImageProcessor.erode_mask(state.mask_current,
                                          self.cfg.erode_size)
        inv = ImageProcessor.invert_mask(state.mask_current)
        state.points_positive = distribute_points(
            eroded,
            self.cfg.num_positive_points,
            n_init=self.cfg.kmeans_n_init,
            random_state=self.cfg.kmeans_random_state)
        state.points_negative = distribute_points(
            inv,
            self.cfg.num_negative_points,
            n_init=self.cfg.kmeans_n_init,
            random_state=self.cfg.kmeans_random_state)
        return state
