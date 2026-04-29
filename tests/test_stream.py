"""
tests/test_stream.py

Unit tests for the virtual camera integration layer.
Uses a mock pyvirtualcam.Camera so no virtual camera driver is needed.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.stream import VirtualCamera


# -- VirtualCamera tests -------------------------------------------------------


class TestVirtualCamera:

    def _make_vcam(self, **kwargs) -> VirtualCamera:
        return VirtualCamera(width=640, height=480, fps=30, **kwargs)

    def test_send_converts_bgr_correctly(self):
        """
        VirtualCamera.send() must not raise and must call cam.send() exactly once
        per frame.
        """
        vcam = self._make_vcam()
        mock_cam = MagicMock()
        mock_cam.__enter__ = MagicMock(return_value=mock_cam)
        mock_cam.__exit__ = MagicMock(return_value=False)

        with patch("pyvirtualcam.Camera", return_value=mock_cam):
            vcam.open()
            vcam._cam = mock_cam
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            vcam.send(frame)
            mock_cam.send.assert_called_once()

    def test_send_resizes_mismatched_frame(self):
        """
        If the frame dimensions don't match the configured resolution,
        send() should resize rather than crash.
        """
        vcam = self._make_vcam()
        mock_cam = MagicMock()
        vcam._cam = mock_cam

        # Send a 360p frame to a 480p camera
        small_frame = np.zeros((360, 640, 3), dtype=np.uint8)
        vcam.send(small_frame)  # should not raise

        # Verify send was called with the resized frame
        sent_frame = mock_cam.send.call_args[0][0]
        assert sent_frame.shape == (480, 640, 3)

    def test_repeat_last_frame_sends_previous(self):
        vcam = self._make_vcam()
        mock_cam = MagicMock()
        vcam._cam = mock_cam

        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        vcam.send(frame)
        mock_cam.reset_mock()

        vcam.repeat_last_frame()
        mock_cam.send.assert_called_once()

    def test_repeat_last_frame_no_op_when_no_frame(self):
        """repeat_last_frame should silently do nothing if no frame has been sent."""
        vcam = self._make_vcam()
        mock_cam = MagicMock()
        vcam._cam = mock_cam
        vcam.repeat_last_frame()  # should not raise
        mock_cam.send.assert_not_called()

    def test_send_raises_when_not_opened(self):
        vcam = self._make_vcam()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="not opened"):
            vcam.send(frame)

    def test_close_is_idempotent(self):
        """close() called twice should not raise."""
        vcam = self._make_vcam()
        mock_cam = MagicMock()
        mock_cam.__exit__ = MagicMock(return_value=False)
        vcam._cam = mock_cam
        vcam.close()
        vcam.close()  # second call should be safe
