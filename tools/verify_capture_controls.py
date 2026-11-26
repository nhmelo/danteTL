"""Record only our fixture's screen rectangle; verify border changes and clean OCR input."""

import argparse
import json
import os
import time
from collections.abc import Callable
from ctypes import wintypes as wt
from pathlib import Path

import shiboken6
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPalette, QPixmap
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication, QWidget

from dantetl.capture.windows import Windows
from dantetl.capture.worker import CaptureSession
from dantetl.domain import CaptureState, Rect, WindowTarget, sample_box
from dantetl.ocr.observation import TextTrack
from dantetl.translation.values import TranslatedText, TranslationRequest
from dantetl.ui.demo import DemoWindow
from dantetl.ui.inspection import frame_image
from dantetl.ui.overlay import OverlayWindow


def verify_menu_blur(
    demo: DemoWindow,
    overlay: OverlayWindow,
    target: WindowTarget,
    windows: Windows,
    client: Rect,
    grab: Callable[[], QImage],
    pump: Callable[[float], None],
    output: Path,
) -> list[int]:
    # Fine stripes distinguish a real backdrop from both a dark fill and bare transparency.
    pattern = QPixmap(8, 8)
    pattern.fill(QColor(220, 220, 220))
    painter = QPainter(pattern)
    painter.fillRect(0, 0, 4, 8, QColor(40, 40, 40))
    painter.end()
    menu = QWidget(demo)
    menu.setGeometry(demo.rect())
    palette = menu.palette()
    palette.setBrush(QPalette.ColorRole.Window, QBrush(pattern))
    menu.setPalette(palette)
    menu.setAutoFillBackground(True)
    menu.show()
    menu.raise_()
    cell_width, cell_height = client.width // 8, client.height // 16
    tracks = tuple(
        TextTrack(
            i + 1,
            "項目",
            Rect(i % 8 * cell_width, i // 8 * cell_height, cell_width - 6, cell_height - 4),
        )
        for i in range(128)
    )
    translations = tuple(TranslatedText(TranslationRequest.from_track(t), "A") for t in tracks)
    overlay.set_text_style(False, 10)
    overlay.set_backdrop_blur(True)

    def stripe_values(image: QImage, box: Rect) -> list[int]:
        y = box.y + box.height // 2
        return [
            image.pixelColor(x, y).red()
            for x in range(box.x + box.width // 2, box.x + box.width - 3)
        ]

    def blurred(values: list[int]) -> bool:
        return max(values) - min(values) < 32 and 75 < sum(values) / len(values) < 180

    counts = [8, 16, 17, 32, 128, 17, 8, 0]
    for count in counts:
        overlay.set_scene(tracks, translations=translations[:count], english=True)
        pump(0.4)
        image = grab()
        deadline = time.monotonic() + 5
        while count and not blurred(stripe_values(image, tracks[count - 1].bounds)):
            assert time.monotonic() < deadline, f"Blur did not reach menu label {count}"
            pump(0.08)
            image = grab()
        for i, track in enumerate(tracks):
            values = stripe_values(image, track.bounds)
            if i < count:
                assert blurred(values), (
                    f"Missing blur at menu label {i + 1} with {count} translations"
                )
            else:
                assert max(values) - min(values) > 120, f"Stale blur at menu label {i + 1}"
        if count == len(tracks):
            image.save(str(output / "dense-menu.png"))
        if count == 16:
            assert overlay._backdrops is not None
            handles = overlay._backdrops.visible_handles
            assert len(handles) == count
            visibility = QSignalSpy(overlay.windowHandle().visibleChanged)
            # Stay within the opaque topmost fixture. An independent striped cover
            # exposes leaked blur as lost contrast and leaked text as changed pixels.
            cover = QWidget(
                None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
            )
            cover.setPalette(palette)
            cover.setAutoFillBackground(True)
            cover.resize(demo.width() // 2, demo.height())
            cover.move(demo.mapToGlobal(QPoint(0, 0)))
            cover.show()
            cover.raise_()
            try:
                overlay.align(client, anchor=target)
                pump(0.5)
                # Force a reorder between the existing labels and blur surfaces.
                assert windows.user.SetWindowPos(
                    int(cover.winId()), int(overlay.winId()), 0, 0, 0, 0, 0x13
                )
                overlay.align(client, anchor=target)
                pump(0.1)
                composed = grab()
                reference = cover.grab().toImage()
                area = tracks[1].bounds
                assert composed.copy(area.x, area.y, area.width, area.height).convertToFormat(
                    QImage.Format.Format_RGB32
                ) == reference.copy(area.x, area.y, area.width, area.height).convertToFormat(
                    QImage.Format.Format_RGB32
                ), "Text or blur leaked onto the covering window"
                assert blurred(stripe_values(composed, tracks[7].bounds))
                windows.user.WindowFromPoint.argtypes = [wt.POINT]
                windows.user.WindowFromPoint.restype = wt.HWND
                for label, expected in ((1, int(cover.winId())), (7, target.hwnd)):
                    box = tracks[label].bounds
                    point = wt.POINT(client.x + box.x + 6, client.y + box.y + 6)
                    assert windows.user.WindowFromPoint(point) == expected, "Overlay blocked input"
                composed.save(str(output / "partially-covered.png"))
            finally:
                cover.close()
            overlay.align(client)
            pump(0.4)
            assert visibility.count() == 0, "Focus changes hid the translation surface"
            assert overlay._backdrops.visible_handles == handles, "Focus changes replaced the blur"
            assert all(windows.user.IsWindowVisible(hwnd) for hwnd in handles)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--menu-blur", action="store_true", help="Verify dense native backdrops")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    windows = Windows()
    windows.enable_dpi_awareness()
    app = QApplication([])
    demo = DemoWindow()
    demo.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
    demo.resize(700, 440)
    demo.move(app.primaryScreen().availableGeometry().center() - demo.rect().center())
    demo.show()
    demo.raise_()
    app.processEvents()
    target = WindowTarget(int(demo.winId()), os.getpid(), demo.windowTitle())
    session = CaptureSession(target, windows, fps=8)
    overlay = OverlayWindow(windows)
    evidence = {}

    def pump(seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)

    def wait_frame(after):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            value = session.snapshot()
            assert value.state != CaptureState.FAILED, value.message
            if value.frame and value.frame.sequence > after:
                return value.frame
            pump(0.05)
        raise AssertionError("No new frame")

    def screen_image():
        # Capture only the rectangle of our opaque, topmost fixture, never a monitor.
        rect = demo.frameGeometry()
        return (
            demo.screen().grabWindow(0, rect.x(), rect.y(), rect.width(), rect.height()).toImage()
        )

    def client_image():
        origin = demo.mapToGlobal(QPoint(0, 0))
        return (
            demo.screen()
            .grabWindow(0, origin.x(), origin.y(), demo.width(), demo.height())
            .toImage()
        )

    def yellow_edge_pixels(image):
        # The fixture's corner marker is inside the client, not this outer 4px rim.
        return sum(
            1
            for y in range(image.height())
            for x in range(image.width())
            if min(x, y, image.width() - x - 1, image.height() - y - 1) < 4
            and (c := image.pixelColor(x, y)).red() > 150
            and c.green() > 130
            and c.blue() < 100
        )

    try:
        session.start()
        first = wait_frame(0)
        overlay.align(first.geometry.client)
        pump(0.5)
        before = screen_image()
        before.save(str(args.output / "border-on.png"))
        current = session.snapshot().frame
        assert current is not None
        sequence = current.sequence
        session.set_border(False)
        assert session.snapshot().frame is not None
        pump(0.8)
        after = wait_frame(sequence)
        assert after.sequence > sequence
        without = screen_image()
        without.save(str(args.output / "border-off.png"))
        box = sample_box(after.width, after.height)
        raw = frame_image(after)
        assert raw.pixelColor(box.x + 5, box.y + 5).name() == "#29465e"
        # Screen composition includes the translucent overlay at the same client pixel.
        client = client_image()
        assert client.width() == after.width and client.height() == after.height
        assert client.pixelColor(box.x + 5, box.y + 5).name() != "#29465e"
        client.save(str(args.output / "screen-with-overlay.png"))
        raw.save(str(args.output / "raw-without-overlay.png"))
        evidence.update(
            recording_contains_overlay=True,
            raw_window_capture_excludes_overlay=True,
            native_affinity=windows.overlay_properties(int(overlay.winId()))[1],
            sequence_before=sequence,
            sequence_after=after.sequence,
            yellow_pixels_before=yellow_edge_pixels(before),
            yellow_pixels_after=yellow_edge_pixels(without),
        )
        hwnd = int(overlay.winId())
        for included in (False, True, False, True):
            overlay.set_capture_included(included)
            pump(0.2)
            assert windows.overlay_properties(hwnd)[1] == (0 if included else 0x11)
            assert overlay.isVisible() and int(overlay.winId()) == hwnd
            recorded = client_image()
            assert (recorded.pixelColor(box.x + 5, box.y + 5).name() != "#29465e") == included
            recorded.save(str(args.output / f"recording-included-{included}.png"))
            current = wait_frame(after.sequence)
            assert frame_image(current).pixelColor(box.x + 5, box.y + 5).name() == "#29465e"
        evidence["recording_toggle_live_without_replacing_overlay_or_capture"] = True
        evidence["exclusion_keeps_clean_game_pixels_in_recording"] = True
        session.set_border(True)
        pump(0.8)
        restored = wait_frame(after.sequence)
        shown = screen_image()
        shown.save(str(args.output / "border-restored.png"))
        evidence["yellow_pixels_restored"] = yellow_edge_pixels(shown)
        evidence["sequence_restored"] = restored.sequence
        evidence["border_visibly_disabled"] = (
            evidence["yellow_pixels_before"] > evidence["yellow_pixels_after"] + 100
            and evidence["yellow_pixels_restored"] > evidence["yellow_pixels_after"] + 100
        )
        assert evidence["native_affinity"] == 0
        assert evidence["border_visibly_disabled"], evidence
        if args.menu_blur:
            evidence["verified_blurred_label_counts"] = verify_menu_blur(
                demo,
                overlay,
                target,
                windows,
                restored.geometry.client,
                client_image,
                pump,
                args.output,
            )
    finally:
        session.request_stop()
        deadline = time.monotonic() + 10
        while not session.finished and time.monotonic() < deadline:
            pump(0.05)
        assert session.finished, "Capture did not drain"
        overlay.close()
        demo.close()
        shiboken6.delete(overlay)
        shiboken6.delete(demo)
        shiboken6.delete(app)
    (args.output / "result.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
