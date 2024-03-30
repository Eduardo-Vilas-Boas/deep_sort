import argparse
import errno
import os
import time

import cv2
import numpy as np
from torchvision.models import MobileNet_V3_Small_Weights

from deep_sort.image_encoder import ImageEncoder


def extract_image_patch(image, bbox, patch_shape):
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.

    """
    bbox = np.array(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int32)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    image = cv2.resize(image, tuple(patch_shape[::-1]))
    return image


def create_box_encoder(model_repo, model_name, model_weights, batch_size=32):
    image_encoder = ImageEncoder(model_repo, model_name, model_weights)
    image_shape = image_encoder.image_shape

    def encoder(image, boxes):
        image_patches = []
        for box in boxes:
            patch = extract_image_patch(image, box, image_shape[:2])
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(0.0, 255.0, image_shape).astype(
                    np.uint8
                )
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        return image_encoder(image_patches, batch_size)

    return encoder


def generate_detections(encoder, mot_dir, output_dir, detection_dir=None):
    """Generate detections with features.

    Parameters
    ----------
    encoder : Callable[image, ndarray] -> ndarray
        The encoder function takes as input a BGR color image and a matrix of
        bounding boxes in format `(x, y, w, h)` and returns a matrix of
        corresponding feature vectors.
    mot_dir : str
        Path to the MOTChallenge directory (can be either train or test).
    output_dir
        Path to the output directory. Will be created if it does not exist.
    detection_dir
        Path to custom detections. The directory structure should be the default
        MOTChallenge structure: `[sequence]/det/det.txt`. If None, uses the
        standard MOTChallenge detections.

    """
    if detection_dir is None:
        detection_dir = mot_dir
    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno == errno.EEXIST and os.path.isdir(output_dir):
            pass
        else:
            raise ValueError(
                "Failed to created output directory '%s'" % output_dir
            )

    print(f"Processing detections - {os.listdir(mot_dir)}")

    for sequence in os.listdir(mot_dir):
        print("Processing %s" % sequence)
        sequence_dir = os.path.join(mot_dir, sequence)

        image_dir = os.path.join(sequence_dir, "img1")
        image_filenames = {
            int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
        }

        detection_file = os.path.join(detection_dir, sequence, "det/det.txt")
        detections_in = np.loadtxt(detection_file, delimiter=",")
        detections_out = []

        frame_indices = detections_in[:, 0].astype(np.int32)
        min_frame_idx = frame_indices.astype(np.int32).min()
        max_frame_idx = frame_indices.astype(np.int32).max()
        max_frame_idx = 50

        print("min_frame_idx", min_frame_idx)
        print("max_frame_idx", max_frame_idx)
        time.sleep(1)

        for frame_idx in range(min_frame_idx, max_frame_idx + 1):
            print("Frame %05d/%05d" % (frame_idx, max_frame_idx))
            mask = frame_indices == frame_idx
            rows = detections_in[mask]

            if frame_idx not in image_filenames:
                print("WARNING could not find image for frame %d" % frame_idx)
                continue
            bgr_image = cv2.imread(
                image_filenames[frame_idx], cv2.IMREAD_COLOR
            )
            image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

            features = encoder(image, rows[:, 2:6].copy())
            detections_out += [
                np.r_[(row, feature)] for row, feature in zip(rows, features)
            ]

        output_filename = os.path.join(output_dir, "%s.npy" % sequence)
        np.save(
            output_filename, np.asarray(detections_out), allow_pickle=False
        )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Re-ID feature extractor")
    parser.add_argument(
        "--model-detection",
        default="resources/networks/yolov8s.pt",
        help="Relative path of ultranalytics model to use.",
    )
    parser.add_argument(
        "--model-repo",
        default="pytorch/vision:v0.17.0",
        help="PyTorch model repository to use.",
    )
    parser.add_argument(
        "--model-name", default="mobilenet_v3_small", help="Model to use."
    )
    parser.add_argument(
        "--mot_dir",
        help="Path to MOTChallenge directory (train or test)",
        required=True,
    )
    parser.add_argument(
        "--detection_dir",
        help="Path to custom detections. Defaults to "
        "standard MOT detections Directory structure should be the default "
        "MOTChallenge structure: [sequence]/det/det.txt",
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        help="Output directory. Will be created if it does not" " exist.",
        default="detections",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.model_name == "mobilenet_v3_small":
        encoder = create_box_encoder(
            args.model_repo,
            args.model_name,
            MobileNet_V3_Small_Weights.DEFAULT,
            batch_size=32,
        )

    generate_detections(
        encoder, args.mot_dir, args.output_dir, args.detection_dir
    )


if __name__ == "__main__":
    main()
