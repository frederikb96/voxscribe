"""Run: python -m unittest discover -s tests"""

import unittest

from voxscribe.providers import MIN_OVERLAP, strip_overlap


class StripOverlapTest(unittest.TestCase):
    def test_full_duplicate_segment_is_dropped(self) -> None:
        existing = "Some earlier text. And I hate this. It's super annoying. Can you fix it?"
        incoming = "And I hate this. It's super annoying. Can you fix it?"
        self.assertEqual(strip_overlap(existing, incoming), "")

    def test_partial_overlap_is_trimmed(self) -> None:
        existing = "We should look at the clipboard truncation problem today"
        incoming = "the clipboard truncation problem today and tomorrow as well"
        self.assertEqual(strip_overlap(existing, incoming), "and tomorrow as well")

    def test_short_repeats_are_kept(self) -> None:
        existing = "Yeah, okay. Yeah."
        incoming = "Yeah. Right."
        self.assertEqual(strip_overlap(existing, incoming), "Yeah. Right.")

    def test_no_overlap_unchanged(self) -> None:
        self.assertEqual(strip_overlap("abc " * 20, "completely different"), "completely different")

    def test_min_overlap_boundary(self) -> None:
        tail = "x" * MIN_OVERLAP
        self.assertEqual(strip_overlap("prefix " + tail, tail + " more"), "more")
        short = "y" * (MIN_OVERLAP - 1)
        self.assertEqual(strip_overlap("prefix " + short, short + " more"), short + " more")


if __name__ == "__main__":
    unittest.main()
