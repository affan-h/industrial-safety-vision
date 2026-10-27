import pytest

from rules import Cooldown, DwellTimer, KofN, SafetyRules, Zone, foot_point, helmet_status, point_in_polygon

SQUARE = [(0, 0), (100, 0), (100, 100), (0, 100)]


@pytest.mark.parametrize("pt,expected", [((50, 50), True), ((150, 50), False), ((-1, 50), False), ((99, 1), True)])
def test_point_in_polygon(pt, expected):
    assert point_in_polygon(pt, SQUARE) is expected


def test_point_in_concave_polygon():
    u_shape = [(0, 0), (30, 0), (30, 70), (70, 70), (70, 0), (100, 0), (100, 100), (0, 100)]
    assert point_in_polygon((50, 30), u_shape) is False  # inside the notch
    assert point_in_polygon((50, 90), u_shape) is True


def test_foot_point_is_bottom_centre():
    assert foot_point((10, 20, 30, 100)) == (20, 100)


def test_helmet_status_three_states():
    person = (100, 100, 200, 400)  # head region: y 100..190
    assert helmet_status(person, helmet_boxes=[(130, 95, 170, 130)], head_boxes=[]) is True
    assert helmet_status(person, helmet_boxes=[], head_boxes=[(130, 100, 170, 140)]) is False
    assert helmet_status(person, helmet_boxes=[], head_boxes=[]) is None
    # a helmet lying near the feet does not count
    assert helmet_status(person, helmet_boxes=[(130, 360, 170, 390)], head_boxes=[(130, 100, 170, 140)]) is False


def test_dwell_accumulates_and_survives_short_exit():
    d = DwellTimer(exit_grace_s=0.5)
    assert d.update(1, True, 0.0) == 0.0
    assert d.update(1, True, 2.0) == 2.0
    d.update(1, False, 2.3)              # 0.3 s flicker outside
    assert d.update(1, True, 2.5) == 2.5  # timer kept
    d.update(1, False, 3.0)
    d.update(1, False, 4.0)              # gone for 1.5 s -> reset
    assert d.update(1, True, 4.1) == 0.0


def test_k_of_n_suppresses_flicker():
    f = KofN(k=3, n=5)
    assert [f.update(7, x) for x in [True, False, True, False, True]] == [False, False, False, False, True]
    assert f.update(8, True) is False  # tracks are independent


def test_cooldown():
    c = Cooldown(10)
    assert c.ready(("a",), 0) and not c.ready(("a",), 5) and c.ready(("b",), 5) and c.ready(("a",), 10.1)


def make_rules(**kw):
    zone = Zone("Bay", SQUARE, "SOP 7.1", max_dwell_s=3)
    args = dict(zones=[zone], helmet_sop="SOP 4.2", k=2, n=3, cooldown_s=10)
    args.update(kw)
    return SafetyRules(**args)


def test_zone_dwell_event_fires_once_after_limit():
    rules = make_rules()
    person = {1: (40, 20, 60, 80)}  # feet at (50, 80): inside
    events = []
    for i in range(0, 50):  # 5 s at 10 fps
        ev, states = rules.update(i, i / 10, person, helmets=[(45, 20, 55, 30)], heads=[])
        events += ev
    assert [e.rule for e in events] == ["zone_dwell"]
    assert events[0].time_s == pytest.approx(3.0)
    assert states[1].zone == "Bay" and states[1].helmet is True


def test_person_standing_outside_is_ignored():
    rules = make_rules()
    events = []
    for i in range(50):
        ev, states = rules.update(i, i / 10, {1: (140, 20, 160, 80)}, [], [])
        events += ev
    assert events == [] and states[1].zone is None


def test_no_helmet_needs_k_of_n_frames():
    rules = make_rules(zones=[])
    person = {3: (300, 100, 400, 400)}
    head = [(330, 100, 370, 140)]
    ev1, _ = rules.update(0, 0.0, person, [], head)
    ev2, st = rules.update(1, 0.1, person, [], head)
    assert ev1 == [] and [e.rule for e in ev2] == ["no_helmet"] and "no_helmet" in st[3].violations


def test_helmet_rule_disabled():
    rules = make_rules(zones=[], helmet_sop=None)
    for i in range(5):
        ev, st = rules.update(i, i / 10, {3: (300, 100, 400, 400)}, [], [(330, 100, 370, 140)])
        assert ev == [] and st[3].helmet is None
