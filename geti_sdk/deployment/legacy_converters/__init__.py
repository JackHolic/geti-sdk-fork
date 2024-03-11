# Copyright (C) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

"""
Prediction converters for use with inference models created in older
versions of the Intel® Geti™ platform, i.e. v1.8 and below.
"""

from .legacy_anomaly_converter import (
    AnomalyClassificationToAnnotationConverter,
    AnomalyDetectionToAnnotationConverter,
    AnomalySegmentationToAnnotationConverter,
)
from .legacy_classification_converter import ClassificationToAnnotationConverter
from .legacy_detection_converter import RotatedRectToAnnotationConverter
from .legacy_segmentation_converter import (
    MaskToAnnotationConverter,
    SegmentationToAnnotationConverter,
)

__all__ = [
    "AnomalyClassificationToAnnotationConverter",
    "AnomalyDetectionToAnnotationConverter",
    "AnomalySegmentationToAnnotationConverter",
    "ClassificationToAnnotationConverter",
    "SegmentationToAnnotationConverter",
    "MaskToAnnotationConverter",
    "RotatedRectToAnnotationConverter",
]
