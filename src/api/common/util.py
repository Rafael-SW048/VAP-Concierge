import time


def nanosleep(interval: int):
    now = time.perf_counter_ns()
    end = now + interval
    while now < end:
        now = time.perf_counter_ns()
