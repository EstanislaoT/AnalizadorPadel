import pytest
import asyncio
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_video_capture():
    """Fixture para simular cv2.VideoCapture"""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 30.0
    mock_cap.read.return_value = (True, MagicMock())
    return mock_cap


@pytest.fixture
def sample_frame():
    """Fixture para simular un frame de video"""
    import numpy as np
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


@pytest.fixture
def temp_output_dir(tmp_path):
    """Fixture para directorio temporal de salida"""
    return tmp_path / "output"


@pytest.fixture(scope="session")
def event_loop():
    """Fixture para el loop de eventos asyncio"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
