"""Values shared at module boundaries; coordinates are physical pixels unless stated."""

from dataclasses import dataclass
from enum import StrEnum

MAX_OVERLAY_LABELS = 128


@dataclass(frozen=True, slots=True)
class Rect:
    x: int
    y: int
    width: int
    height: int

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    def contains(self, other: "Rect") -> bool:
        return (
            self.x <= other.x
            and self.y <= other.y
            and self.x + self.width >= other.x + other.width
            and self.y + self.height >= other.y + other.height
        )


def union_bounds(rectangles: tuple[Rect, ...]) -> Rect:
    """Smallest physical rectangle containing a nonempty set of rectangles."""
    x, y = min(r.x for r in rectangles), min(r.y for r in rectangles)
    right = max(r.x + r.width for r in rectangles)
    bottom = max(r.y + r.height for r in rectangles)
    return Rect(x, y, right - x, bottom - y)


@dataclass(frozen=True, slots=True)
class WindowTarget:
    hwnd: int
    process_id: int
    title: str


@dataclass(frozen=True, slots=True)
class WindowGeometry:
    client: Rect
    visible_bounds: Rect
    window_bounds: Rect
    dpi: int
    minimized: bool = False


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    sequence: int
    captured_at: float
    geometry: WindowGeometry
    bgra: bytes

    @property
    def width(self) -> int:
        return self.geometry.client.width

    @property
    def height(self) -> int:
        return self.geometry.client.height


class CaptureState(StrEnum):
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CaptureSnapshot:
    state: CaptureState
    frame: CapturedFrame | None = None
    message: str = ""


def client_crop(frame_size: tuple[int, int], geometry: WindowGeometry) -> Rect | None:
    """Reconcile WGC's observed size with known bounds; never guess during resize.

    WGC usually supplies DWM visible bounds, while some targets supply their client
    area or whole window. A matching size plus containment determines a safe crop.
    """
    if geometry.minimized or min(geometry.client.size) <= 0:
        return None
    for bounds in (geometry.visible_bounds, geometry.window_bounds, geometry.client):
        if bounds.size == frame_size and bounds.contains(geometry.client):
            return Rect(
                geometry.client.x - bounds.x,
                geometry.client.y - bounds.y,
                geometry.client.width,
                geometry.client.height,
            )
    return None


def sample_box(width: int, height: int) -> Rect:
    """A fixed proportional dialogue region for milestone-one alignment checks."""
    return Rect(
        round(width * 0.08), round(height * 0.72), round(width * 0.84), round(height * 0.19)
    )


def overlay_ready(
    snapshot: CaptureSnapshot,
    geometry: WindowGeometry | None,
    focus_allows_overlay: bool,
) -> bool:
    """Require a current-size frame and the selected overlay visibility policy."""
    return bool(
        snapshot.state == CaptureState.RUNNING
        and snapshot.frame is not None
        and geometry is not None
        and not geometry.minimized
        and focus_allows_overlay
        and snapshot.frame.geometry.client.size == geometry.client.size
    )
