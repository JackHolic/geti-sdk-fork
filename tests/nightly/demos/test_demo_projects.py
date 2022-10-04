# Copyright (C) 2022 Intel Corporation
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
from typing import Tuple

import pytest
from _pytest.fixtures import FixtureRequest

from geti_sdk import Geti
from geti_sdk.annotation_readers import DatumAnnotationReader
from geti_sdk.demos import ensure_trained_anomaly_project, get_coco_dataset
from geti_sdk.demos.data_helpers.anomaly_helpers import is_ad_dataset
from geti_sdk.demos.data_helpers.coco_helpers import (
    COCOSubset,
    directory_has_coco_subset,
)
from geti_sdk.demos.demo_projects import ensure_project_is_trained
from geti_sdk.demos.demo_projects.coco_demos import (
    DEMO_LABELS,
    DEMO_PROJECT_NAME,
    DEMO_PROJECT_TYPE,
)
from geti_sdk.rest_clients import (
    AnnotationClient,
    ImageClient,
    PredictionClient,
    ProjectClient,
)
from tests.helpers import SdkTestMode, force_delete_project
from tests.helpers.constants import PROJECT_PREFIX


class TestDemoProjects:
    def test_get_coco_dataset(self, fxt_coco_dataset: str):
        """
        Test that the `get_coco_dataset` method returns the path to a directory
        containing the val2017 subset of the coco dataset.
        """
        assert directory_has_coco_subset(fxt_coco_dataset, COCOSubset.VAL2017)

    def test_get_mvtec_dataset(self, fxt_anomaly_dataset: str):
        """
        Test that the `get_mvtec_dataset` method returns the path to a directory
        containing the MVTec AD 'transistor' dataset
        """
        assert is_ad_dataset(fxt_anomaly_dataset)

    @pytest.mark.parametrize(
        "demo_project_fixture_name",
        [
            "fxt_classification_demo_project",
            "fxt_anomaly_classification_demo_project",
            "fxt_segmentation_demo_project",
            "fxt_detection_to_classification_demo_project",
            "fxt_detection_to_segmentation_demo_project",
        ],
        ids=[
            "Classification",
            "Anomaly classification",
            "Segmentation",
            "Detection to classification",
            "Detection to segmentation",
        ],
    )
    def test_create_demo_projects(
        self,
        request,
        demo_project_fixture_name: str,
        fxt_geti_no_vcr: Geti,
        fxt_project_client_no_vcr: ProjectClient,
        fxt_demo_images_and_annotations: Tuple[int, int],
    ):
        project = request.getfixturevalue(demo_project_fixture_name)
        project_on_server = fxt_project_client_no_vcr.get_project_by_name(project.name)
        image_client = ImageClient(
            session=fxt_geti_no_vcr.session,
            workspace_id=fxt_geti_no_vcr.workspace_id,
            project=project_on_server,
        )
        annotation_client = AnnotationClient(
            session=fxt_geti_no_vcr.session,
            workspace_id=fxt_geti_no_vcr.workspace_id,
            project=project_on_server,
        )
        images = image_client.get_all_images()
        annotations = [annotation_client.get_annotation(image) for image in images]

        assert len(images) == fxt_demo_images_and_annotations[0]
        assert len(annotations) == fxt_demo_images_and_annotations[1]
        for attribute_name in ["name", "pipeline", "creation_time", "id", "creator_id"]:
            assert getattr(project_on_server, attribute_name) == getattr(
                project, attribute_name
            )

    @pytest.mark.vcr()
    def test_ensure_project_is_trained(
        self,
        request: FixtureRequest,
        fxt_geti_no_vcr: Geti,
        fxt_project_client_no_vcr: ProjectClient,
        fxt_test_mode: SdkTestMode,
    ):
        """
        Test that the `ensure_project_is_trained` function results in a trained project
        """
        project_name = f"{PROJECT_PREFIX}_{DEMO_PROJECT_NAME}"
        coco_path = get_coco_dataset()

        # Create annotation reader
        annotation_reader = DatumAnnotationReader(
            base_data_folder=coco_path, annotation_format="coco"
        )
        annotation_reader.filter_dataset(labels=DEMO_LABELS, criterion="OR")

        project = fxt_geti_no_vcr.create_single_task_project_from_dataset(
            project_name=project_name,
            project_type=DEMO_PROJECT_TYPE,
            path_to_images=coco_path,
            annotation_reader=annotation_reader,
            labels=DEMO_LABELS,
            number_of_images_to_upload=12,
            number_of_images_to_annotate=12,
            enable_auto_train=False,
        )
        request.addfinalizer(
            lambda: force_delete_project(project_name, fxt_project_client_no_vcr)
        )
        prediction_client = PredictionClient(
            session=fxt_geti_no_vcr.session,
            workspace_id=fxt_geti_no_vcr.workspace_id,
            project=project,
        )
        assert not prediction_client.ready_to_predict

        ensure_project_is_trained(geti=fxt_geti_no_vcr, project=project)

        assert prediction_client.ready_to_predict

    def test_ensure_trained_anomaly_project(
        self, fxt_geti_no_vcr: Geti, fxt_project_client_no_vcr: ProjectClient
    ):
        """
        Test the `ensure_trained_anomaly_project` method
        """
        project_name = f"{PROJECT_PREFIX}_ensure_trained_anomaly_project"
        if fxt_project_client_no_vcr.get_project_by_name(project_name) is not None:
            force_delete_project(
                project_name=project_name, project_client=fxt_project_client_no_vcr
            )
        assert project_name not in [
            project.name for project in fxt_project_client_no_vcr.get_all_projects()
        ]

        project = ensure_trained_anomaly_project(
            geti=fxt_geti_no_vcr, project_name=project_name
        )
        prediction_client = PredictionClient(
            session=fxt_geti_no_vcr.session,
            workspace_id=fxt_geti_no_vcr.workspace_id,
            project=project,
        )
        assert prediction_client.ready_to_predict