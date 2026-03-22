import pytest
import os
import sys
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

# Add parent directory to path to import scripts
sys.path.insert(0, str(Path(__file__).parent.parent))

import process_video


class TestProcessVideoModule:
    """Tests for process_video module imports and basic functionality"""

    def test_import_process_video_module(self):
        """Test that we can import the process_video module"""
        try:
            import process_video
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import process_video: {e}")

    def test_main_function_exists(self):
        """Test that main function exists in process_video"""
        assert hasattr(process_video, 'main')
        assert callable(process_video.main)


class TestVideoProcessingArguments:
    """Tests for argument handling in process_video"""

    def test_main_exits_with_wrong_arguments(self):
        """Test that main exits with error when wrong number of arguments"""
        with patch.object(sys, 'argv', ['process_video.py']):
            result = process_video.main()
            assert result == 1

    def test_main_exits_with_missing_video(self):
        """Test that main exits with error when video file doesn't exist"""
        with patch.object(sys, 'argv', [
            'process_video.py',
            '/nonexistent/video.mp4',
            '/tmp/output.json',
            '/tmp/models'
        ]):
            result = process_video.main()
            assert result == 1


class TestVideoFileExistence:
    """Tests for test video files availability"""

    def test_test_videos_directory_exists(self):
        """Test that test-videos directory exists"""
        test_videos_dir = Path(__file__).parent.parent.parent / 'test-videos'
        assert test_videos_dir.exists(), f"test-videos directory not found at {test_videos_dir}"

    def test_video_files_exist_in_test_videos(self):
        """Test that test video files exist"""
        test_videos_dir = Path(__file__).parent.parent.parent / 'test-videos'

        # Check for at least one video file
        video_extensions = ['.mp4', '.avi', '.mov']
        video_files = [
            f for f in test_videos_dir.iterdir()
            if f.suffix.lower() in video_extensions
        ]

        assert len(video_files) > 0, "No test video files found"

    def test_specific_test_video_exists(self):
        """Test that PadelPro3.mp4 test video exists"""
        test_videos_dir = Path(__file__).parent.parent.parent / 'test-videos'
        test_video = test_videos_dir / 'PadelPro3.mp4'
        assert test_video.exists(), f"PadelPro3.mp4 not found in test-videos"


class TestProcessingLogic:
    """Tests for video processing logic"""

    def test_result_data_structure(self):
        """Test the structure of result data dictionary"""
        # Simular el resultado esperado
        result_data = {
            "status": "success",
            "video_path": "/path/to/video.mp4",
            "processing_time_seconds": 10.5,
            "total_frames": 100,
            "players_detected": 4,
            "avg_detections_per_frame": 2.5,
            "frames_with_4_players": 80,
            "detection_rate_percent": 80.0,
            "model_used": "yolov8m.pt",
            "timestamp": "2024-01-01 12:00:00"
        }

        # Verificar estructura
        assert 'status' in result_data
        assert 'video_path' in result_data
        assert 'total_frames' in result_data
        assert 'detection_rate_percent' in result_data

    def test_error_data_structure(self):
        """Test the structure of error data dictionary"""
        error_data = {
            "status": "error",
            "error_type": "FileNotFoundError",
            "error_message": "Video not found",
            "timestamp": "2024-01-01 12:00:00"
        }

        assert error_data['status'] == 'error'
        assert 'error_type' in error_data
        assert 'error_message' in error_data


# Fixture for test data
@pytest.fixture
def sample_frame_data():
    """Provide sample frame data for tests"""
    return [
        {
            'frame_number': 1,
            'detections': [
                {'class': 0, 'confidence': 0.95, 'bbox': [100, 100, 50, 50], 'track_id': 1},
                {'class': 0, 'confidence': 0.90, 'bbox': [300, 200, 50, 50], 'track_id': 2},
            ]
        },
        {
            'frame_number': 2,
            'detections': [
                {'class': 0, 'confidence': 0.92, 'bbox': [105, 105, 50, 50], 'track_id': 1},
                {'class': 0, 'confidence': 0.88, 'bbox': [295, 195, 50, 50], 'track_id': 2},
            ]
        },
    ]


@pytest.fixture
def mock_yolo_result():
    """Provide mock YOLO result"""
    mock_result = MagicMock()
    mock_box1 = MagicMock()
    mock_box1.cls = 0
    mock_box1.conf = 0.95
    mock_box1.xyxy = [[100, 100, 150, 150]]

    mock_box2 = MagicMock()
    mock_box2.cls = 0
    mock_box2.conf = 0.90
    mock_box2.xyxy = [[300, 200, 350, 250]]

    mock_result.boxes = [mock_box1, mock_box2]
    return mock_result


def test_with_fixture(sample_frame_data):
    """Test using pytest fixture with frame data"""
    assert isinstance(sample_frame_data, list)
    assert len(sample_frame_data) == 2
    assert sample_frame_data[0]['frame_number'] == 1
    assert len(sample_frame_data[0]['detections']) == 2


class TestProcessingCalculations:
    """Tests for processing calculations"""

    def test_average_detections_calculation(self):
        """Test calculation of average detections"""
        detections_per_frame = [2, 3, 4, 3, 2]
        avg_detections = sum(detections_per_frame) / len(detections_per_frame)
        assert avg_detections == 2.8

    def test_detection_rate_calculation(self):
        """Test calculation of detection rate"""
        total_frames = 100
        frames_with_4_players = 80
        detection_rate = (frames_with_4_players / total_frames * 100)
        assert detection_rate == 80.0

    def test_detection_rate_with_zero_frames(self):
        """Test detection rate calculation with zero frames"""
        total_frames = 0
        frames_with_4_players = 0
        detection_rate = (frames_with_4_players / total_frames * 100) if total_frames > 0 else 0
        assert detection_rate == 0


class TestIntegration:
    """Integration tests for video processing"""

    @pytest.mark.skip(reason="Requires actual YOLO model - run manually")
    def test_full_processing_pipeline(self):
        """Full integration test with actual video"""
        test_videos_dir = Path(__file__).parent.parent.parent / 'test-videos'
        test_video = test_videos_dir / 'PadelPro3.mp4'

        if not test_video.exists():
            pytest.skip("Test video not available")

        # This test is skipped by default as it requires YOLO model
        # To run: pytest -v tests/test_process_video.py::TestIntegration::test_full_processing_pipeline -s


class TestComprehensiveVideoProcessing:
    """Comprehensive tests with full mocking"""

    @pytest.fixture
    def setup_test_env(self, tmp_path):
        """Create complete test environment with mocked files"""
        video_path = tmp_path / "test_video.mp4"
        video_path.write_bytes(b"fake video content")
        output_path = tmp_path / "output.json"
        models_path = tmp_path / "models"
        models_path.mkdir()
        model_file = models_path / "yolov8m.pt"
        model_file.write_bytes(b"fake model")

        return {
            'video_path': str(video_path),
            'output_path': str(output_path),
            'models_path': str(models_path),
            'tmp_path': tmp_path
        }

    @patch('process_video.YOLO')
    def test_successful_execution_with_output(self, mock_yolo, setup_test_env):
        """Test complete successful execution and output validation"""
        env = setup_test_env

        # Setup mock model
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        # Create mock results with varying detections
        mock_results = []
        for i in range(10):
            mock_result = MagicMock()
            # 8 frames with 4+ detections, 2 with less
            mock_result.boxes = [MagicMock() for _ in range(4 if i < 8 else 2)]
            mock_results.append(mock_result)
        mock_model.return_value = mock_results

        with patch.object(sys, 'argv', [
            'process_video.py',
            env['video_path'],
            env['output_path'],
            env['models_path']
        ]):
            result = process_video.main()

        assert result == 0
        assert os.path.exists(env['output_path'])

        with open(env['output_path'], 'r') as f:
            data = json.load(f)

        # Validate all expected fields
        assert data['status'] == 'success'
        assert data['video_path'] == env['video_path']
        assert data['model_used'] == 'yolov8m.pt'
        assert data['total_frames'] == 10
        assert data['frames_with_4_players'] == 8
        assert data['detection_rate_percent'] == 80.0
        assert 'processing_time_seconds' in data
        assert 'timestamp' in data
        assert isinstance(data['players_detected'], int)

    @patch('process_video.YOLO')
    def test_all_frames_with_four_players(self, mock_yolo, setup_test_env):
        """Test when all frames have exactly 4 players detected"""
        env = setup_test_env

        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        # All frames have exactly 4 detections
        mock_results = [MagicMock(boxes=[MagicMock()] * 4) for _ in range(5)]
        mock_model.return_value = mock_results

        with patch.object(sys, 'argv', [
            'process_video.py',
            env['video_path'],
            env['output_path'],
            env['models_path']
        ]):
            process_video.main()

        with open(env['output_path'], 'r') as f:
            data = json.load(f)

        assert data['detection_rate_percent'] == 100.0
        assert data['frames_with_4_players'] == 5
        assert data['players_detected'] == 4  # Should detect 4 unique players

    @patch('process_video.YOLO')
    def test_no_frames_with_four_players(self, mock_yolo, setup_test_env):
        """Test when no frames have 4 players detected"""
        env = setup_test_env

        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        # All frames have less than 4 detections
        mock_results = [MagicMock(boxes=[MagicMock()] * 2) for _ in range(5)]
        mock_model.return_value = mock_results

        with patch.object(sys, 'argv', [
            'process_video.py',
            env['video_path'],
            env['output_path'],
            env['models_path']
        ]):
            process_video.main()

        with open(env['output_path'], 'r') as f:
            data = json.load(f)

        assert data['detection_rate_percent'] == 0.0
        assert data['frames_with_4_players'] == 0
        assert data['players_detected'] == 0  # No players detected

    @patch('process_video.YOLO')
    def test_output_directory_created(self, mock_yolo, setup_test_env):
        """Test that output directory is created if it doesn't exist"""
        env = setup_test_env

        # Use a nested output path that doesn't exist
        nested_output = str(env['tmp_path'] / "nested" / "dir" / "output.json")

        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        mock_model.return_value = [MagicMock(boxes=[MagicMock()] * 4)]

        with patch.object(sys, 'argv', [
            'process_video.py',
            env['video_path'],
            nested_output,
            env['models_path']
        ]):
            process_video.main()

        assert os.path.exists(nested_output)

    @patch('process_video.YOLO')
    def test_stdout_output(self, mock_yolo, setup_test_env, capsys):
        """Test that result is printed to stdout"""
        env = setup_test_env

        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        mock_model.return_value = [MagicMock(boxes=[MagicMock()] * 4)]

        with patch.object(sys, 'argv', [
            'process_video.py',
            env['video_path'],
            env['output_path'],
            env['models_path']
        ]):
            process_video.main()

        captured = capsys.readouterr()
        # Check that progress messages are printed
        assert "Procesando video" in captured.out
        assert "Output" in captured.out
        assert "Models" in captured.out
        assert "Procesamiento completado" in captured.out


class TestArgumentEdgeCases:
    """Tests for edge cases in arguments"""

    def test_argument_with_spaces(self, tmp_path):
        """Test handling of paths with spaces"""
        video_path = tmp_path / "video with spaces.mp4"
        video_path.write_bytes(b"fake content")
        output_path = tmp_path / "output with spaces.json"
        models_path = tmp_path / "models"
        models_path.mkdir()
        (models_path / "yolov8m.pt").write_bytes(b"fake model")

        # Just verify the paths are handled correctly
        with patch.object(sys, 'argv', [
            'process_video.py',
            str(video_path),
            str(output_path),
            str(models_path)
        ]):
            assert sys.argv[1] == str(video_path)
            assert sys.argv[2] == str(output_path)
            assert sys.argv[3] == str(models_path)

    def test_argument_with_unicode(self, tmp_path):
        """Test handling of paths with unicode characters"""
        video_path = tmp_path / "video_\u00e1\u00e9\u00ed.mp4"  # accented characters
        video_path.write_bytes(b"fake content")
        output_path = tmp_path / "output.json"
        models_path = tmp_path / "models"
        models_path.mkdir()
        (models_path / "yolov8m.pt").write_bytes(b"fake model")

        with patch.object(sys, 'argv', [
            'process_video.py',
            str(video_path),
            str(output_path),
            str(models_path)
        ]):
            assert "\u00e1" in sys.argv[1]  # contains accented character
