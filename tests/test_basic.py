import os
from rumor_detection.data import load_dataset
from rumor_detection.models import _make_pipeline

def test_imports():
    assert callable(load_dataset)
    assert callable(_make_pipeline)
