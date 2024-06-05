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
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
IMG_FORMATS = 'jpeg', 'jpg'
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#ROOT = os.path.dirname(__file__)

def get_top_level_parent_directory(file_path, levels):
    parent_directory = file_path
    for n in range(levels):
        parent_directory = os.path.dirname(parent_directory)
    return parent_directory  

def run(source = ROOT / "images",
        dest = ROOT / "dest",
        save_pass_images = False,
        model = None,
        stain_min_area = 0,
        ltim_min_area = 0):
    stain_min_area = 0
    ltim_min_area = 0
    save_pass_images = True
    #Create offline deployment
    offline_deployment = Deployment.from_folder(model)
    #Load model
    offline_deployment.load_inference_models(device="CPU")
    #Load image files
    files = []
    if os.path.isdir(source):
        files.extend(sorted(glob.glob(os.path.join(source,'*.*'))))
    else:
        raise FileNotFoundError(f"{source} does not exist.")
    imagefiles = [file for file in files if file.split('.')[1].lower() in IMG_FORMATS]
    for image in imagefiles:
        im = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        im = CLAHE.apply(im) # Improve image contrast.
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        t_start = time.time()
        pred = offline_deployment.infer(im_rgb)
        t_elapsed = time.time() - t_start
        save_image_with_annotation_scene(im_rgb,pred,filepath=image, destination=dest, save_pass_images=save_pass_images, inferencing_time=t_elapsed, stain_min_area=stain_min_area, ltim_min_area=ltim_min_area)
def save_image_with_annotation_scene(
    image: Union[Image, VideoFrame, np.ndarray],
    annotation_scene: Union[AnnotationScene, Prediction],
    filepath: Optional[str] = None,
    destination: str = None,
    save_pass_images: bool = False,
    channel_order: str = "rgb",
    show_labels: bool = True,
    show_confidences: bool = True,
    fill_shapes: bool = True,
    inferencing_time : float = 0,
    stain_min_area : int = 0,
    ltim_min_area : int = 0
):
    """
    Save an image with an annotation_scene overlayed on top of it.

    :param image: Image to show prediction for.
    :param annotation_scene: Annotations or Predictions to overlay on the image
    :param filepath: Optional filepath to save the image with annotation overlay to.
        If left as None, the result will not be saved to file
    :param show_in_notebook: True if the image needs to be shown in a notebook context.
        Setting this to True will display the image inline in the notebook. Setting it
        to False will open a pop up to show the image.
    :param show_results: True to show the results. If `show_in_notebook` is True, this
        will display the image with the annotations inside the notebook. If
        `show_in_notebook` is False, a new opencv window will pop up. If
        `show_results` is set to False, the results will not be shown but will only
        be returned instead
    :param channel_order: The channel order (R,G,B or B,G,R) used for the input image.
        This parameter accepts either `rgb` or `bgr` as input values, and defaults to
        `rgb`.
    :param show_labels: True to show the labels of the annotations. If set to False,
        the labels will not be shown.
    :param show_confidences: True to show the confidences of the annotations. If set to
        False, the confidences will not be shown.
    :param fill_shapes: True to fill the shapes of the annotations. If set to False, the
        shapes will not be filled.
    """
    if not os.path.exists(destination):
        os.mkdir(destination)
    p = Path(filepath)
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
        show_confidence=show_confidences,
    )
    # ote_annotation_scene = annotation_scene.to_ote(
    #     image_width=media_information.width, image_height=media_information.height
    # )

    if isinstance(image, np.ndarray):
        numpy_image = image.copy()
    else:
        numpy_image = image.numpy.copy()

    if channel_order == "bgr":
        rgb_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
    elif channel_order == "rgb":
        rgb_image = numpy_image
    else:
        raise ValueError(
            f"Invalid channel order '{channel_order}'. Please use either `rgb` or "
            f"`bgr`."
        )
    #result = visualizer.draw(image=rgb_image, annotation=ote_annotation_scene)
    result = visualizer.draw(
        image=rgb_image, annotation=annotation_scene, fill_shapes=fill_shapes
    )
    # area_list = ote_annotation_scene.area
    annotations = [x for x in annotation_scene.annotations]
    failmode_list = []
    unique_failmode = []
    num_detect = 0
    area_list = []
    for annotation in annotations:
        failmode = annotation.labels[0].name
        area = int(annotation.shape.area)
        if "stain" in str.lower(failmode):
            if area >= stain_min_area:
                num_detect += 1
                print(f'Detected stain area = {str(area)} is bigger than min area {stain_min_area}. Save to final list. Final list = {str(num_detect)}')
            else:
                print(print(f'Detected stain area = {str(area)} is smaller than min area {stain_min_area}. Do not save to final list. Final list = {str(num_detect)}'))
        elif "ltim" in str.lower(failmode):
            if area >= ltim_min_area:
                num_detect += 1
                print(f'Detected LTIM area = {str(area)} is bigger than min area {ltim_min_area}. Save to final list. Final list = {str(num_detect)}')
            else:
                print(print(f'Detected LTIM area = {str(area)} is smaller than min area {ltim_min_area}. Do not save to final list. Final list = {str(num_detect)}'))
        elif "damage" in str.lower(failmode) or "scratch" in str.lower(failmode) or "pedestal" in str.lower(failmode):
            num_detect += 1
        failmode_list.append(failmode)
        area_list.append(area)
    [unique_failmode.append(item) for item in failmode_list if item not in unique_failmode]
    #failmode_list = list(set(failmode_list))
    finalarea = (('_R').join((str(area_list)).split(',')))[1:-1].replace(' ','')
    finallabel = ('_').join(unique_failmode)
    if num_detect != 0 and len(failmode_list):
        savepath = destination + '/' + 'Fail_' + finallabel + f"_R{finalarea}" + '_' + p.name
    elif num_detect == 0 and len(failmode_list):
        savepath = destination + '/' + 'Pass_' + finallabel + f"_RC{finalarea}" + '_' + p.name
    else:
        savepath = destination + '/' + 'Pass_' + p.name

    if save_pass_images and 'Pass' in savepath:
        cv2.imwrite(savepath,result)
        print(f"Saved pass image at {savepath}. Inferencing time {inferencing_time*1000:.2f} ms")
    if 'Fail' in savepath and "NoRead" not in savepath:
        cv2.imwrite(savepath,result)
        print(f'Saved fail image at {savepath}. Inferencing time {inferencing_time*1000:.2f} ms')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',type=str,default= 'C:\\Users\\dangkho1\\OneDrive - Intel Corporation\\Documents\\MAE\\AI\\Geti\\latest_geti\\geti-sdk\\geti_sdk\\demos\\data\\1444', help="source of input images")
    parser.add_argument('--dest',type=str, default= "C:\\Users\\dangkho1\\OneDrive - Intel Corporation\\Documents\\MAE\\AI\\Geti\\latest_geti\\geti-sdk\\runs\\1444", help="destination of output")
    parser.add_argument('--save_pass_images', action='store_true',help="save pass image")
    parser.add_argument('--model', type=str, default="C:\\Users\\dangkho1\\OneDrive - Intel Corporation\\Documents\\MAE\\AI\\Geti\\latest_geti\\geti-sdk\\notebooks\\deployments\\bare",help="model checkpoint")
    parser.add_argument('--stain_min_area', type=int, default=0, help='Define min area to detect stain defects.')
    parser.add_argument('--ltim_min_area', type=int, default=0, help='Define min area for LTIM defects.')
    opt = parser.parse_args()
    return opt
def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)



