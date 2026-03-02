import importlib.util
import sys
import types
import unittest
import warnings
from pathlib import Path

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


class DegenerateBoxReproTests(unittest.TestCase):
    def test_process_avoids_divide_by_zero_runtime_warning(self):
        class NumpyRow:
            def __init__(self, arr):
                self._arr = arr

            def numpy(self):
                return self._arr

        class FakeBox:
            def __init__(self):
                # x1 == x2 means width==0 in MY_CV.py aspect-ratio logic.
                self.xyxy = [NumpyRow(np.array([10.0, 10.0, 10.0, 40.0], dtype=np.float32))]

        class FakeBoxes(list):
            def cpu(self):
                return self

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


if __name__ == "__main__":
    unittest.main()
