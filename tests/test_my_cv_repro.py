import importlib.util
import os
import re
import sys
import types
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MY_CV_PATH = REPO_ROOT / "gcv" / "MY_CV.py"


def load_module_with_ultralytics(yolo_class):
    fake_ultralytics = types.ModuleType("ultralytics")
    fake_ultralytics.YOLO = yolo_class
    sys.modules["ultralytics"] = fake_ultralytics

    spec = importlib.util.spec_from_file_location("my_cv_under_test", str(MY_CV_PATH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NumpyRow:
    def __init__(self, arr):
        self._arr = arr

    def numpy(self):
        return self._arr


class FakeBoxes(list):
    def cpu(self):
        return self


def decode_rel(packet):
    rel_x = int.from_bytes(packet[1:3], byteorder="big", signed=True)
    rel_y = int.from_bytes(packet[3:5], byteorder="big", signed=True)
    return rel_x, rel_y


class ModelPathAndDeviceReproTests(unittest.TestCase):
    def test_model_path_points_to_repo_models_weight(self):
        class StubYOLO:
            def __init__(self, model_path):
                self.model_path = model_path

            def to(self, device):
                return self

        module = load_module_with_ultralytics(StubYOLO)
        expected = (REPO_ROOT / "models" / "apex_8n.pt").resolve()
        actual = Path(module.MODEL_PATH).resolve()

        self.assertEqual(
            actual,
            expected,
            f"MODEL_PATH should resolve to {expected}, got {actual}",
        )
        self.assertTrue(expected.exists(), "Expected model file is missing from models/")

    def test_worker_falls_back_to_cpu_when_mps_unavailable(self):
        class MPSUnavailableYOLO:
            def __init__(self, model_path):
                self.model_path = model_path
                self.last_device = None

            def to(self, device):
                self.last_device = device
                if device == "mps":
                    raise RuntimeError("MPS backend is not available")
                return self

        module = load_module_with_ultralytics(MPSUnavailableYOLO)

        worker = module.GCVWorker(1920, 1080)

        self.assertEqual(
            worker.model.last_device,
            "cpu",
            f"Expected CPU fallback device, got {worker.model.last_device!r}",
        )


class StartupPreflightReproTests(unittest.TestCase):
    def test_startup_preflight_raises_clear_error_when_no_model_exists(self):
        missing_gcv = (REPO_ROOT / "gcv" / "apex_8n.pt").resolve()
        missing_models = (REPO_ROOT / "models" / "apex_8n.pt").resolve()
        real_exists = os.path.exists

        def exists_with_models_missing(path):
            resolved = Path(path).resolve()
            if resolved in {missing_gcv, missing_models}:
                return False
            return real_exists(path)

        class ShouldNotConstructYOLO:
            def __init__(self, model_path):
                raise AssertionError("YOLO should not construct when startup preflight fails")

        error_re = (
            rf"(?i)model.*not found.*"
            rf"{re.escape(str(missing_gcv))}.*"
            rf"{re.escape(str(missing_models))}"
        )

        with self.assertRaisesRegex(FileNotFoundError, error_re):
            with mock.patch("os.path.exists", side_effect=exists_with_models_missing):
                module = load_module_with_ultralytics(ShouldNotConstructYOLO)
                module.GCVWorker(1920, 1080)


class DegenerateBoxReproTests(unittest.TestCase):
    def test_process_avoids_divide_by_zero_runtime_warning(self):
        class FakeBox:
            def __init__(self):
                # x1 == x2 means width==0 in MY_CV.py aspect-ratio logic.
                self.xyxy = [NumpyRow(np.array([10.0, 10.0, 10.0, 40.0], dtype=np.float32))]

        class DegenerateYOLO:
            def __init__(self, model_path):
                self.model_path = model_path

            def to(self, device):
                return self

            def predict(self, **kwargs):
                result = types.SimpleNamespace()
                result.boxes = FakeBoxes([FakeBox()])
                return [result]

        module = load_module_with_ultralytics(DegenerateYOLO)
        worker = module.GCVWorker(640, 480)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            out_frame, gcvdata = worker.process(frame)

        self.assertEqual(out_frame.shape, frame.shape)
        self.assertEqual(len(gcvdata), 5)


class GCVDataSmokeDecodeTests(unittest.TestCase):
    def test_smoke_decodes_signed_rel_vector(self):
        class FakeBox:
            def __init__(self):
                # center_x=320, center_y=240 for 640x480; rel_y resolves to -12 on first frame.
                self.xyxy = [NumpyRow(np.array([300.0, 200.0, 340.0, 280.0], dtype=np.float32))]

        class DeterministicYOLO:
            def __init__(self, model_path):
                self.model_path = model_path

            def to(self, device):
                return self

            def predict(self, **kwargs):
                result = types.SimpleNamespace()
                result.boxes = FakeBoxes([FakeBox()])
                return [result]

        module = load_module_with_ultralytics(DeterministicYOLO)
        worker = module.GCVWorker(640, 480)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        _, gcvdata = worker.process(frame)

        self.assertEqual(len(gcvdata), 5)
        self.assertEqual(gcvdata[0], 1)
        self.assertEqual(list(gcvdata), [1, 0, 0, 255, 244])

        rel_x = int.from_bytes(gcvdata[1:3], byteorder="big", signed=True)
        rel_y = int.from_bytes(gcvdata[3:5], byteorder="big", signed=True)
        self.assertEqual(rel_x, 0)
        self.assertEqual(rel_y, -12)


class UserReportProcessReproTests(unittest.TestCase):
    def test_process_does_not_propagate_predict_runtime_error(self):
        class PredictErrorYOLO:
            def __init__(self, model_path):
                self.model_path = model_path

            def to(self, device):
                return self

            def predict(self, **kwargs):
                raise RuntimeError("Error 121: MPS command buffer failure")

        module = load_module_with_ultralytics(PredictErrorYOLO)
        worker = module.GCVWorker(640, 480)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        try:
            _, gcvdata = worker.process(frame)
        except RuntimeError as exc:
            self.fail(f"process() should not propagate predict exceptions, got: {exc}")

        self.assertEqual(list(gcvdata), [1, 0, 0, 0, 0])

    def test_process_treats_none_boxes_as_empty_detections(self):
        class NoneBoxesYOLO:
            def __init__(self, model_path):
                self.model_path = model_path

            def to(self, device):
                return self

            def predict(self, **kwargs):
                result = types.SimpleNamespace()
                result.boxes = None
                return [result]

        module = load_module_with_ultralytics(NoneBoxesYOLO)
        worker = module.GCVWorker(640, 480)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        try:
            _, gcvdata = worker.process(frame)
        except TypeError as exc:
            self.fail(f"process() should treat boxes=None as no detections, got: {exc}")

        self.assertEqual(list(gcvdata), [1, 0, 0, 0, 0])

    def test_process_skips_nan_box_coordinates(self):
        class NaNBox:
            def __init__(self):
                self.xyxy = [NumpyRow(np.array([np.nan, 10.0, np.nan, 40.0], dtype=np.float32))]

        class NaNBoxYOLO:
            def __init__(self, model_path):
                self.model_path = model_path

            def to(self, device):
                return self

            def predict(self, **kwargs):
                result = types.SimpleNamespace()
                result.boxes = FakeBoxes([NaNBox()])
                return [result]

        module = load_module_with_ultralytics(NaNBoxYOLO)
        worker = module.GCVWorker(640, 480)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        try:
            _, gcvdata = worker.process(frame)
        except ValueError as exc:
            self.fail(f"process() should skip NaN detections instead of crashing, got: {exc}")

        self.assertEqual(list(gcvdata), [1, 0, 0, 0, 0])

    def test_process_resets_vector_immediately_after_target_loss(self):
        class TargetBox:
            def __init__(self):
                self.xyxy = [NumpyRow(np.array([300.0, 200.0, 340.0, 280.0], dtype=np.float32))]

        class TargetThenNoneYOLO:
            def __init__(self, model_path):
                self.model_path = model_path
                self.calls = 0

            def to(self, device):
                return self

            def predict(self, **kwargs):
                self.calls += 1
                result = types.SimpleNamespace()
                result.boxes = FakeBoxes([TargetBox()]) if self.calls == 1 else FakeBoxes([])
                return [result]

        module = load_module_with_ultralytics(TargetThenNoneYOLO)
        worker = module.GCVWorker(640, 480)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        _, first = worker.process(frame.copy())
        _, second = worker.process(frame.copy())

        _, first_rel_y = decode_rel(first)
        _, second_rel_y = decode_rel(second)

        self.assertNotEqual(first_rel_y, 0, "Setup check: first frame should move aim")
        self.assertEqual(
            second_rel_y,
            0,
            f"Expected immediate neutral vector after target loss, got rel_y={second_rel_y}",
        )

    def test_process_keeps_small_distant_target_candidate(self):
        class SmallFarBox:
            def __init__(self):
                # h=20 (< old 25 cutoff), w=10, ratio=2.0; should still be considered.
                self.xyxy = [NumpyRow(np.array([315.0, 180.0, 325.0, 200.0], dtype=np.float32))]

        class SmallFarYOLO:
            def __init__(self, model_path):
                self.model_path = model_path

            def to(self, device):
                return self

            def predict(self, **kwargs):
                result = types.SimpleNamespace()
                result.boxes = FakeBoxes([SmallFarBox()])
                return [result]

        module = load_module_with_ultralytics(SmallFarYOLO)
        worker = module.GCVWorker(640, 480)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        _, gcvdata = worker.process(frame)
        _, rel_y = decode_rel(gcvdata)
        self.assertNotEqual(rel_y, 0, "Expected small target to produce non-zero correction")

    def test_process_rejects_oversized_box_candidate(self):
        class OversizedBox:
            def __init__(self):
                # Large area candidate that should be rejected as unstable/false-positive.
                self.xyxy = [NumpyRow(np.array([210.0, 40.0, 430.0, 360.0], dtype=np.float32))]

        class OversizedYOLO:
            def __init__(self, model_path):
                self.model_path = model_path

            def to(self, device):
                return self

            def predict(self, **kwargs):
                result = types.SimpleNamespace()
                result.boxes = FakeBoxes([OversizedBox()])
                return [result]

        module = load_module_with_ultralytics(OversizedYOLO)
        worker = module.GCVWorker(640, 480)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        _, gcvdata = worker.process(frame)
        _, rel_y = decode_rel(gcvdata)
        self.assertEqual(rel_y, 0, f"Expected oversized candidate rejection, got rel_y={rel_y}")


if __name__ == "__main__":
    unittest.main()
