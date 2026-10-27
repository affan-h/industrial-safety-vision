from prepare_data import remap_label_lines, split_names


def test_remap_keeps_only_target_classes():
    lines = [
        "0 0.5 0.5 0.2 0.4",   # person -> 0
        "12 0.5 0.2 0.05 0.05",  # head -> 1
        "10 0.5 0.1 0.06 0.05",  # helmet -> 2
        "16 0.5 0.5 0.1 0.2",   # safety-vest -> 3
        "3 0.4 0.4 0.01 0.01",  # face -> dropped
        "",                     # blank -> dropped
    ]
    assert remap_label_lines(lines) == [
        "0 0.5 0.5 0.2 0.4",
        "1 0.5 0.2 0.05 0.05",
        "2 0.5 0.1 0.06 0.05",
        "3 0.5 0.5 0.1 0.2",
    ]


def test_split_is_deterministic_and_disjoint():
    names = [f"img{i}.jpg" for i in range(100)]
    train, val = split_names(names, 0.1, seed=0)
    assert len(val) == 10 and len(train) == 90
    assert not set(train) & set(val)
    assert (train, val) == split_names(list(reversed(names)), 0.1, seed=0)
