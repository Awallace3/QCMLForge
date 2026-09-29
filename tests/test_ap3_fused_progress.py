from apnet_pt.pt_datasets.ap3_fused_ds import _progress_percent_milestones


def test_progress_milestones_continue_from_resumed_cursor():
    total_rows = 1_509_251
    resumed_cursor = 1_218_864

    assert list(
        _progress_percent_milestones(resumed_cursor, 1_222_500, total_rows)
    ) == [81]
    assert list(
        _progress_percent_milestones(1_222_500, 1_509_251, total_rows)
    ) == list(range(82, 101))


def test_progress_milestones_handle_empty_and_clamp_to_100_percent():
    assert list(_progress_percent_milestones(0, 10, 0)) == []
    assert list(_progress_percent_milestones(99, 101, 100)) == [100]
