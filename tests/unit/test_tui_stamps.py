"""The timestamps on a chat line have to mean something.

A replayed conversation used to be stamped with the time it was *loaded*,
so yesterday's work opened looking like it had just happened — every line
in the history said the same minute, the one the app started in.
"""

from datetime import datetime, timedelta

from tokirecur.tui.widgets import _format_stamp


def test_todays_lines_carry_a_clock_time():
    stamp = _format_stamp()
    assert ":" in stamp
    assert len(stamp) == 8, stamp  # HH:MM:SS


def test_a_replayed_line_keeps_the_moment_it_happened():
    """The stamp is the stored time, not the time the app started."""
    # Anchored to today: an hour ago is *yesterday* between midnight and
    # one in the morning, and a line from yesterday carries its day, not
    # a bare clock time.
    when = datetime.now().replace(hour=0, minute=0, second=1).timestamp()
    stamp = _format_stamp(when)
    assert stamp == datetime.fromtimestamp(when).strftime("%H:%M:%S")


def test_yesterday_says_so():
    when = (datetime.now() - timedelta(days=1)).timestamp()
    stamp = _format_stamp(when)
    assert stamp.startswith("yesterday "), stamp


def test_an_older_day_carries_its_date():
    when = (datetime.now() - timedelta(days=6)).timestamp()
    stamp = _format_stamp(when)
    assert "yesterday" not in stamp
    assert datetime.fromtimestamp(when).strftime("%d %b") in stamp, stamp
