import cv2
import numpy as np
from geti_sdk.deployment import Deployment
from typing import List, Optional, Union
#from geti_sdk import Geti
import os
import glob
import argparse
import cv2
import time
from geti_sdk.data_models.annotation_scene import AnnotationScene
from geti_sdk.data_models.predictions import Prediction
from geti_sdk.data_models.enums import AnnotationKind
#from geti_sdk.utils import show_image_with_annotation_scene
from IPython.display import display
from geti_sdk.prediction_visualization.visualizer import Visualizer
import numpy as np
from pathlib import Path
from geti_sdk.data_models.media import Image, VideoFrame
from PIL import Image as PILImage
from geti_sdk.deployment import Deployment
# Load the pre-trained MobileNetV2 model
model = r'C:\Users\dangkho1\OneDrive - Intel Corporation\Documents\MAE\AI\Geti\geti-sdk-fork\deployment\video'
offline_deployment = Deployment.from_folder(model)
#Load model
offline_deployment.load_inference_models(device="CPU")

def run_inference_for_single_image(
    model: str,
    image:Union[Image, VideoFrame, np.ndarray],
    annotation_scene: Union[AnnotationScene, Prediction],
    show_labels: bool = True,
    show_confidences: bool = True,
    fill_shapes: bool = True
):
    if annotation_scene.kind == AnnotationKind.ANNOTATION:
        plot_type = "Annotation"
    elif annotation_scene.kind == AnnotationKind.PREDICTION:
        plot_type = "Prediction"
    else:
        raise ValueError(
            f"Invalid input: Unable to plot object of type {type(annotation_scene)}."
        )
    if isinstance(image, np.ndarray):
    # media_information = MediaInformation(
    #     "", height=image.shape[0], width=image.shape[1]
    # )
        name = "Numpy image"
    else:
        #media_information = image.media_information
        name = image.name

    window_name = f"{plot_type} for {name}"
    visualizer = Visualizer(
    window_name=window_name,
    show_labels=show_labels,
    show_confidence=show_confidences)

    result = visualizer.draw(
    image=image, annotation=annotation_scene, fill_shapes=fill_shapes)
    return result
# Access the webcam
cap = cv2.VideoCapture(0)  # 0 is the default camera

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to stop.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Preprocess the frame
    #input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #input_frame = np.expand_dims(input_frame, axis=0)
    pred = offline_deployment.infer(frame)
    framed = run_inference_for_single_image(model,frame,pred)

    # Display the annotated frame
    cv2.imshow('Live Object Detection', framed)
    
    # Press 'q' to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
