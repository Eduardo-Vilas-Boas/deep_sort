from __future__ import absolute_import, division, print_function

import argparse
import time

import cv2
import numpy as np
import streamlit as st
from torchvision.models import MobileNet_V3_Small_Weights

import streamlit_app
from deep_sort import nn_matching
from deep_sort.image_encoder import ImageEncoder
from deep_sort.image_object_detector import ImageObjectDetector
from deep_sort.tracker import Tracker


def draw_frame_tracks(image_placeholder, frame, results):
    # Draw each bounding box on the frame
    for result in results:
        bbox = np.array(
            [result[0], result[1], result[2], result[3]],
            dtype=np.int32,
        )

        color = (0, 255, 0)  # Green
        cv2.rectangle(
            frame,
            (bbox[0], bbox[1]),
            (bbox[0] + bbox[2], bbox[1] + bbox[3]),
            color,
            2,
        )

    # Display the frame
    image_placeholder.image(frame, channels="BGR")

    # Wait for 30 ms
    time.sleep(0.03)


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

    # Create a placeholder for the image
    image_placeholder = st.empty()

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

            # Draw the bounding boxes on the frame
            draw_frame_tracks(image_placeholder, frame, results)

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


if __name__ == "__main__":
    # Create a form
    with st.form(key="my_form"):
        sequence = st.text_input(
            label="Enter sequence", value="resources/videos/car_traffic.mp4"
        )
        output_file = st.text_input(
            label="Enter output file", value="resources/detections/output.csv"
        )
        min_confidence = st.number_input(
            label="Enter minimum confidence", value=0.5
        )
        nms_max_overlap = st.number_input(
            label="Enter NMS max overlap", value=1.0
        )
        min_detection_height = st.number_input(
            label="Enter minimum detection height", value=0
        )
        max_cosine_distance = st.number_input(
            label="Enter max cosine distance", value=0.2
        )
        nn_budget = st.number_input(label="Enter NN budget", value=100)
        model_detection = st.text_input(
            label="Enter model detection",
            value="resources/networks/yolov8s.pt",
        )
        model_repo = st.text_input(
            label="Enter model repo", value="pytorch/vision:v0.17.0"
        )
        model_name = st.text_input(
            label="Enter model name", value="mobilenet_v3_small"
        )

        # Create a submit button
        submit_button = st.form_submit_button(label="RUN")

    # When the user clicks the "RUN" button, the form is submitted and this code is run
    if submit_button:
        # Here you can call your function to run the algorithms and display the video frames
        # Make sure to pass the input values from the form to your function
        run(
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
        )
