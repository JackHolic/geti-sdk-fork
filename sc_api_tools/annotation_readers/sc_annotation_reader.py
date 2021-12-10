import glob
import json
import os
from typing import List, Dict, Optional

from .base_annotation_reader import AnnotationReader


class SCAnnotationReader(AnnotationReader):
    def __init__(
            self,
            base_data_folder: str,
            annotation_format: str = ".json",
            task_type: str = "detection",
            task_type_to_label_names_mapping: Optional[Dict[str, List[str]]] = None
    ):
        if annotation_format != '.json':
            raise ValueError(
                f"Annotation format {annotation_format} is currently not"
                f" supported by the SCAnnotationReader"
            )
        super().__init__(
            base_data_folder=base_data_folder,
            annotation_format=annotation_format,
            task_type=task_type
        )
        self._task_label_mapping = task_type_to_label_names_mapping

    def _get_labels_for_current_task_type(self, all_labels: List[str]) -> List[str]:
        """
        Returns the labels for the task type the annotation reader is currently set to.

        :param all_labels: List of all label names in the project
        """
        if self._task_label_mapping is None:
            print("No label mapping defined, including all labels")
            labels = all_labels
        else:
            labels = [
                label for label in all_labels
                if label in self._task_label_mapping[self.task_type]
            ]
        return labels

    def get_data(self, filename: str, label_name_to_id_mapping: dict):
        filepath = glob.glob(
            os.path.join(
                self.base_folder, f"{filename}{self.annotation_format}")
        )
        if len(filepath) > 1:
            print(
                f"Multiple matching annotation files found for image with "
                f"name {filename}. Skipping this image..."
            )
            return []
        elif len(filepath) == 0:
            print(
                f"No matching annotation file found for image with name {filename}."
                f" Skipping this image..."
            )
            return []
        else:
            filepath = filepath[0]
        with open(filepath, 'r') as f:
            data = json.load(f)
        task_labels = self._get_labels_for_current_task_type(
            list(label_name_to_id_mapping.keys()))
        new_annotations = []
        for annotation in data["annotations"]:
            labels = annotation["labels"]
            label_names = [label["name"] for label in labels]
            for label in labels:
                label["id"] = label_name_to_id_mapping[label["name"]]
            if all([label in task_labels for label in label_names]):
                new_annotations.append(annotation)
        return new_annotations

    def get_all_label_names(self) -> List[str]:
        """
        Retrieve the unique label names for all annotations in the annotation folder

        :return: List of label names
        """
        print(f"Reading annotation files in folder {self.base_folder}...")
        unique_label_names = []
        for annotation_file in os.listdir(self.base_folder):
            with open(os.path.join(self.base_folder, annotation_file), 'r') as f:
                data = json.load(f)
            for annotation in data["annotations"]:
                labels = [label["name"] for label in annotation["labels"]]
                for label in labels:
                    if label not in unique_label_names:
                        unique_label_names.append(label)
        return unique_label_names