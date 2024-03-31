from __future__ import absolute_import, division, print_function

import argparse

import cv2
from torchvision.models import MobileNet_V3_Small_Weights

from deep_sort import nn_matching
from deep_sort.image_encoder import ImageEncoder
from deep_sort.image_object_detector import ImageObjectDetector
from deep_sort.tracker import Tracker


def run(
    sequence,
    output_file,
    min_confidence,
    nms_max_overlap,
    min_detection_height,
    max_cosine_distance,
    nn_budget,
    model_detection,
    model_repo,
    model_name,
):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    data : str
        Video sequence to track.
    sequence : str
        Video sequence to track.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    model_detection : str
        Path to the ultralytics model to use.
    model_repo : str
        PyTorch model repository to use.
    model_name : str
        Model to use.
    """

    weights = None
    if model_name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT

    image_encoder = ImageEncoder(model_repo, model_name, weights)
    image_object_detector = ImageObjectDetector(
        model_path=model_detection,
        image_encoder=image_encoder,
        min_confidence=min_confidence,
        min_height=min_detection_height,
        nms_max_overlap=nms_max_overlap,
    )

    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget
    )
    tracker = Tracker(metric)

    # Open the video file
    cap = cv2.VideoCapture(sequence)

    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")

    i = 0
    while cap.isOpened():
        print(f"Frame {i}/{total_frames}")
        # Read a frame from the video
        ret, frame = cap.read()

        results = []
        if ret:
            # Convert the frame to a PIL image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Load image and generate detections.
            detections = image_object_detector(image)

            # Update tracker.
            tracker.predict()
            tracker.update(detections)

            # Store results.
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlwh()

                results.append(
                    [
                        track.track_id,
                        bbox[0],
                        bbox[1],
                        bbox[2],
                        bbox[3],
                    ]
                )

            # Store results.
            f = open(output_file, "a")
            for row in results:
                print(
                    f"{0}, {row[0]}, {row[1]}, {row[2]}, {row[3]}, {row[4]}",
                    file=f,
                )
        else:
            break
        i += 1

    # Release the video file
    cap.release()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--sequence",
        help="Video sequence to track.",
        default=None,
        required=True,
    )
    parser.add_argument(
        "--output_file",
        help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="resources/detections/hypotheses.txt",
    )
    parser.add_argument(
        "--min_confidence",
        help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.8,
        type=float,
    )
    parser.add_argument(
        "--min_detection_height",
        help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--nms_max_overlap",
        help="Non-maxima suppression threshold: Maximum " "detection overlap.",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--max_cosine_distance",
        help="Gating threshold for cosine distance "
        "metric (object appearance).",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--nn_budget",
        help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.",
        type=int,
        default=None,
    )
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.sequence,
        args.output_file,
        args.min_confidence,
        args.nms_max_overlap,
        args.min_detection_height,
        args.max_cosine_distance,
        args.nn_budget,
        args.model_detection,
        args.model_repo,
        args.model_name,
    )
