"""Safety rule logic, kept free of OpenCV/YOLO so it can be unit-tested.

Boxes are (x1, y1, x2, y2) in pixels. Times are in seconds.
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field


def point_in_polygon(pt: tuple[float, float], poly: list[tuple[float, float]]) -> bool:
    """Ray-casting test."""
    x, y = pt
    inside = False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def foot_point(box) -> tuple[float, float]:
    """Bottom-centre of a person box: where they stand on the floor.

    On a fixed, elevated camera the box centre can sit over a zone drawn on the
    floor while the person is actually standing outside it; the feet don't lie.
    """
    x1, _, x2, y2 = box
    return ((x1 + x2) / 2, y2)


def head_region(box, frac: float = 0.3):
    x1, y1, x2, y2 = box
    return (x1, y1, x2, y1 + (y2 - y1) * frac)


def _center_in(box, region) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]


def helmet_status(person_box, helmet_boxes, head_boxes):
    """True = helmet seen on this person, False = head seen but no helmet on it, None = can't tell.

    The head class covers every head, helmeted or not (in SH17 81% of helmets sit
    on a labelled head), so a head only counts as bare when no helmet is found.

    "Can't tell" (back turned, head occluded, too far away) is NOT treated as a
    violation; alerting on it would flood supervisors with false alarms.
    """
    region = head_region(person_box)
    if any(_center_in(h, region) for h in helmet_boxes):
        return True
    if any(_center_in(h, region) for h in head_boxes):
        return False
    return None


class DwellTimer:
    """Seconds each track has spent continuously inside a zone.

    Only being *seen outside* for longer than `exit_grace_s` resets the timer, so
    a box flickering across the edge or a person briefly occluded keeps their time.
    """

    def __init__(self, exit_grace_s: float = 0.5):
        self.exit_grace_s = exit_grace_s
        self.entered: dict[int, float] = {}
        self.left_at: dict[int, float] = {}

    def update(self, track_id: int, inside: bool, t: float) -> float:
        if inside:
            self.left_at.pop(track_id, None)
            self.entered.setdefault(track_id, t)
            return t - self.entered[track_id]
        if track_id in self.entered:
            left = self.left_at.setdefault(track_id, t)
            if t - left > self.exit_grace_s:
                del self.entered[track_id], self.left_at[track_id]
        return 0.0


class KofN:
    """Confirms a condition only if it held in at least k of the last n frames."""

    def __init__(self, k: int, n: int):
        self.k, self.n = k, n
        self.history: dict[int, deque] = defaultdict(lambda: deque(maxlen=self.n))

    def update(self, track_id: int, flag: bool) -> bool:
        h = self.history[track_id]
        h.append(flag)
        return sum(h) >= self.k


class Cooldown:
    """At most one alert per (track, rule) every `seconds`."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self.last: dict[tuple, float] = {}

    def ready(self, key: tuple, t: float) -> bool:
        if key in self.last and t - self.last[key] < self.seconds:
            return False
        self.last[key] = t
        return True


@dataclass
class Zone:
    name: str
    polygon: list[tuple[float, float]]  # pixels
    sop: str
    max_dwell_s: float


@dataclass
class Event:
    time_s: float
    frame: int
    track_id: int
    rule: str
    zone: str
    sop: str
    detail: str


@dataclass
class TrackState:
    zone: str | None = None
    dwell_s: float = 0.0
    helmet: bool | None = None
    violations: list[str] = field(default_factory=list)


class SafetyRules:
    """Applies the zone-dwell and helmet rules to one frame of tracked detections."""

    def __init__(self, zones: list[Zone], helmet_sop: str | None, k: int, n: int,
                 cooldown_s: float, exit_grace_s: float = 0.5):
        self.zones = zones
        self.helmet_sop = helmet_sop
        self.timers = {z.name: DwellTimer(exit_grace_s) for z in zones}
        self.no_helmet = KofN(k, n)
        self.cooldown = Cooldown(cooldown_s)

    def update(self, frame: int, t: float, people: dict[int, tuple], helmets: list, heads: list):
        events, states = [], {}
        for tid, box in people.items():
            st = TrackState()
            feet = foot_point(box)
            for z in self.zones:
                inside = point_in_polygon(feet, z.polygon)
                dwell = self.timers[z.name].update(tid, inside, t)
                if inside:
                    st.zone, st.dwell_s = z.name, dwell
                if dwell >= z.max_dwell_s:
                    st.violations.append("zone_dwell")
                    if self.cooldown.ready((tid, "zone_dwell", z.name), t):
                        events.append(Event(t, frame, tid, "zone_dwell", z.name, z.sop,
                                            f"in zone for {dwell:.1f}s (limit {z.max_dwell_s:g}s)"))

            if self.helmet_sop:
                st.helmet = helmet_status(box, helmets, heads)
                if self.no_helmet.update(tid, st.helmet is False):
                    st.violations.append("no_helmet")
                    if self.cooldown.ready((tid, "no_helmet", ""), t):
                        events.append(Event(t, frame, tid, "no_helmet", st.zone or "", self.helmet_sop,
                                            f"no helmet on head in {self.no_helmet.k}+ of last {self.no_helmet.n} frames"))
            states[tid] = st
        return events, states
