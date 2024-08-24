import time
from statistics import mean, median, stdev
from typing import Iterable, Self


class TimedIterator[T]:
    """
    An iterator that times the duration of each iteration, printing statistics at end.

    Note:
        Timings in milliseconds

    Example:

        for i in TimedIterator(range(10)):
            # some expensive operation

    """

    def __init__(self, iterable: Iterable[T]) -> None:
        self.iterable = iter(iterable)
        self.timings = []

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> T:
        start_time = time.perf_counter_ns()
        try:
            item = next(self.iterable)
        except StopIteration:
            self._print_statistics()
            raise
        end_time = time.perf_counter_ns()

        duration_ns = end_time - start_time
        duration_ms = 1e-6 * duration_ns
        self.timings.append(duration_ms)
        return item

    def _print_statistics(self):
        if not self.timings:
            print("No iterations performed.")
            return

        total_time = sum(self.timings)
        avg_time = mean(self.timings)
        med_time = median(self.timings)
        min_time = min(self.timings)
        max_time = max(self.timings)
        std_dev = stdev(self.timings) if len(self.timings) > 1 else 0

        unit_str = "ms"

        output_str = "Timing(" + ", ".join(
            [
                f"total={total_time}{unit_str}",
                f"avg={avg_time}{unit_str}",
                # f"median: {med_time}{unit_str}",
                # f"min: {min_time}{unit_str}",
                # f"max: {max_time}{unit_str}",
                # f"std_dev: {std_dev}{unit_str}",
            ]
        )

        output_str += ")"

        print(output_str)
