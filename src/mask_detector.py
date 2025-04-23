import sys
import os
import logging
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from src.SAM_wrapper import SamPredictorWrapper
from src.utility import get_click_coordinates, ImagePathUtility, ImageProcessor

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
    :param mask: Binary mask (numpy array) where points should be distributed.
    :param num_points: Number of points to distribute.
    :param n_init: Number of initializations for KMeans clustering.
    :param random_state: Random state for reproducibility.
    :return: Array of [x, y] coordinates for the distributed points.
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
    Configuration for MaskDetector behavior.
    """
    def __init__(self):
        self.model_type = "vit_h"
        self.checkpoint_path = self._get_resource_path(r"models/sam_vit_h.pth")
        self.is_display = True
        self.downscale_factor = 5
        self.image_extensions = ('.tiff', '.tif', '.png', '.jpg', '.jpeg', '.bmp')
        self.folderpath_source = None
        self.folderpath_save = None
        self.num_positive_points = 2
        self.num_negative_points = 12
        self.is_roi = False
        self.box_roi = None
        self.init_points_positive = None
        self.erode_size = 5
        self.kmeans_n_init = 10
        self.kmeans_random_state = 0

    def _get_resource_path(self, relative_path: str) -> str:
        base = getattr(sys, 'frozen', False) and sys._MEIPASS or os.path.abspath('.')
        return os.path.join(base, relative_path)


class MaskDetectorBuilder:
    """Builder for MaskDetectorConfig"""
    def __init__(self):
        self._cfg = MaskDetectorConfig()

    def set_model_type(self, t: str): self._cfg.model_type = t; return self
    def set_checkpoint(self, p: str): self._cfg.checkpoint_path = p; return self
    def set_display(self, flag: bool): self._cfg.is_display = flag; return self
    def set_downscale(self, f: float): self._cfg.downscale_factor = f; return self
    def set_extensions(self, exts: tuple): self._cfg.image_extensions = exts; return self
    def set_source(self, path: str): self._cfg.folderpath_source = path; return self
    def set_save(self, path: str): self._cfg.folderpath_save = path; return self
    def set_positive_points(self, n: int): self._cfg.num_positive_points = n; return self
    def set_negative_points(self, n: int): self._cfg.num_negative_points = n; return self
    def set_roi(self, flag: bool): self._cfg.is_roi = flag; return self
    def set_box_roi(self, box: tuple): self._cfg.box_roi = box; return self
    def set_init_points(self, pts: np.ndarray): self._cfg.init_points_positive = pts; return self
    # New setters
    def set_erode_size(self, size: int): self._cfg.erode_size = size; return self
    def set_kmeans_n_init(self, n: int): self._cfg.kmeans_n_init = n; return self
    def set_kmeans_random_state(self, r: int): self._cfg.kmeans_random_state = r; return self

    def build(self) -> 'MaskDetector':
        return MaskDetector(self._cfg)


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
        # Avoid ambiguous truth value by explicit None check
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
    def __init__(self, cfg: MaskDetectorConfig = None):
        # if no config is provided, use default
        if cfg is None:
            cfg = MaskDetectorConfig()
        self.cfg = cfg
        self.logger = global_logger
        try:
            self.logger.info('Initializing SAM predictor with model %s', cfg.model_type)
            self._sam = SamPredictorWrapper(cfg.model_type, cfg.checkpoint_path)
        except Exception as e:
            self.logger.error('Failed to load SAM model: %s', e)
            raise

    def process_images(self) -> None:
        try:
            paths = ImagePathUtility.get_image_paths(self.cfg.folderpath_source,
                                                     self.cfg.image_extensions)
        except Exception as e:
            self.logger.error('Error fetching image paths: %s', e)
            return
        os.makedirs(self.cfg.folderpath_save, exist_ok=True)

        # Initial setup
        try:
            first = ImageProcessor.load_image(paths[0])
            first = ImageProcessor.rescale(first, 1/self.cfg.downscale_factor)
        except Exception as e:
            self.logger.error('Cannot load first image: %s', e)
            return

        if self.cfg.is_roi and not self.cfg.box_roi:
            self.cfg.box_roi = self._choose_roi_box()
        if self.cfg.box_roi:
            first = ImageProcessor.crop_image(first, self.cfg.box_roi)

        if self.cfg.init_points_positive is None:
            if not self.cfg.is_display:
                self.logger.error('Headless mode requires init_points_positive')
                return
            self.logger.info('Select %d positive points', self.cfg.num_positive_points)
            pts = get_click_coordinates(cv2.cvtColor(first, cv2.COLOR_RGB2BGR))
        else:
            pts = self.cfg.init_points_positive

        state = ImageProcessingState(pos=pts)

        for path in tqdm(paths, desc='Processing'):
            try:
                img = ImageProcessor.load_image(path)
                img = ImageProcessor.rescale(img, 1/self.cfg.downscale_factor)
                if self.cfg.box_roi:
                    img_proc = ImageProcessor.crop_image(img, self.cfg.box_roi)
                else:
                    img_proc = img

                state = self.process_single_image(img_proc, state)
                # Prepare output mask
                if self.cfg.box_roi:
                    full = np.zeros(img.shape[:2], dtype=np.uint8)
                    x,y,w,h = self.cfg.box_roi
                    full[y:y+h, x:x+w] = state.mask_current
                    out_mask = full
                else:
                    out_mask = state.mask_current

                out_path = self._get_save_path(path)
                ImagePathUtility.save_mask_as_image(out_mask, out_path)
            except Exception as e:
                self.logger.error('Error in processing %s: %s', path, e)
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
        state.points_positive = distribute_points(eroded,
                                                 self.cfg.num_positive_points,
                                                 n_init=self.cfg.kmeans_n_init,
                                                 random_state=self.cfg.kmeans_random_state)
        state.points_negative = distribute_points(inv,
                                                 self.cfg.num_negative_points,
                                                 n_init=self.cfg.kmeans_n_init,
                                                 random_state=self.cfg.kmeans_random_state)
        return state

    def _get_save_path(self, img_path: str) -> str:
        _, fn = os.path.split(img_path)
        name, ext = os.path.splitext(fn)
        return os.path.join(self.cfg.folderpath_save, f"{name}_mask{ext}")

    def _choose_roi_box(self) -> tuple:
        paths = ImagePathUtility.get_image_paths(self.cfg.folderpath_source,
                                                 self.cfg.image_extensions)
        img = ImageProcessor.load_image(paths[0])
        img = ImageProcessor.rescale(img, 1/self.cfg.downscale_factor)
        box = cv2.selectROI("Choose ROI", img, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()
        return box
