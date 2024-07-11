from geti_sdk.deployment import Deployment
import os
import glob
import cv2
import time
from geti_sdk.data_models.predictions import Prediction
from geti_sdk.prediction_visualization.visualizer import Visualizer
import numpy as np
from pathlib import Path
IMG_FORMATS = 'jpeg', 'jpg'
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
def run(source = ROOT / 'inputs',
        dest = ROOT / 'outputs',
        model = ROOT / 'model'):
        #Create offline deployment
        offiline_deployment = Deployment.from_folder(model)
        #Load model
        offiline_deployment.load_inference_models(device='CPU')
        files = []
        if os.path.isdir(source):
                files.extend(sorted(glob.glob(os.path.join(source,'*.*'))))
        else:
                raise FileNotFoundError(f"{source} does not exist")
        imagefiles = [file for file in files if file.split('.')[1].lower() in IMG_FORMATS]
        for image in imagefiles:
                img = cv2.imread(image)
                img_rbg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                pred = offiline_deployment.infer(img_rbg)
                inferencing_images(image=img_rbg,prediction=pred,imagepath=image,destination=dest)
def inferencing_images(
            image: np.ndarray,
            prediction: Prediction,
            imagepath: str = '',
            destination: str = '',
            show_labels: bool = True,
            show_confidence: bool = True,
            fill_shapes: bool = True,
):
        visualizer = Visualizer(
                show_labels=show_labels,
                show_confidence=show_confidence
        )
        if not os.path.exists(destination):
                os.makedirs(destination)
        result = visualizer.draw(
                image=image,
                annotation=prediction,
                fill_shapes=fill_shapes
        )
        savepath = os.path.join(destination,os.path.basename(imagepath))
        cv2.imwrite(savepath,result)
        print(f"Saved image at {savepath}.")
def main():
        run()
if __name__== '__main__':
        main()