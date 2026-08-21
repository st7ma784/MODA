"""Timer-driven ADC capture into a fixed-size ring buffer.

The buffer and every index are preallocated because `push()` runs inside a
MicroPython soft-IRQ timer callback, where heap allocation raises
`MemoryError`. Nothing in this module allocates after `__init__`.

`Ring` and `counts_to_volts` are plain Python and are exercised on CPython by
``pico/tests/test_sampler.py``; only `AdcSource` touches `machine`.
"""

from array import array

ADC_PINS = {0: 26, 1: 27, 2: 28}    # ADC channel -> GPIO number
ADC_FULL_SCALE = 65535              # read_u16() is left-aligned to 16 bits


def counts_to_volts(counts, full_scale_volts):
    return counts * full_scale_volts / ADC_FULL_SCALE


class Ring:
    """Fixed-capacity ring of uint16 ADC counts with a monotonic sample count.

    `total` counts every sample ever pushed, so a reader can ask for
    "everything after sample N" without holding a cursor object; if it falls
    more than `capacity` behind, `since()` tells it how much it missed.
    """

    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError('capacity must be >= 1')
        self.capacity = capacity
        self.buf = array('H', bytearray(2 * capacity))
        self.index = 0
        self.total = 0

    def push(self, value):
        """IRQ context: no allocation, no exceptions."""
        self.buf[self.index] = value
        self.index += 1
        if self.index >= self.capacity:
            self.index = 0
        self.total += 1

    def available(self):
        """Number of samples currently retained (<= capacity)."""
        return self.total if self.total < self.capacity else self.capacity

    def since(self, mark, limit=None):
        """Return `(next_mark, dropped, samples)` for samples after `mark`.

        `dropped` is how many samples were overwritten before the reader got to
        them - non-zero means the consumer is too slow and the trace it draws
        will have a gap.
        """
        available = self.available()
        oldest = self.total - available
        if mark < oldest:
            dropped = oldest - mark
            mark = oldest
        else:
            dropped = 0
        count = self.total - mark
        if limit is not None and count > limit:
            # Skip ahead rather than backlogging: a live trace wants the newest
            # samples, not a slowly-draining queue of stale ones.
            dropped += count - limit
            mark = self.total - limit
            count = limit
        start = (self.index - (self.total - mark)) % self.capacity
        out = []
        for offset in range(count):
            out.append(self.buf[(start + offset) % self.capacity])
        return mark + count, dropped, out

    def latest(self, count=None):
        """The most recent `count` samples, oldest first."""
        available = self.available()
        if count is None or count > available:
            count = available
        start = (self.index - count) % self.capacity
        return [self.buf[(start + offset) % self.capacity] for offset in range(count)]


class AdcSource:
    """Samples one ADC channel at a fixed rate into a `Ring`.

    Reconfiguring rate or channel means tearing the timer down and rebuilding,
    which `reconfigure()` does; the ring is reallocated only when its size
    actually changes, to avoid churning the heap on every settings save.
    """

    def __init__(self, channel, sample_rate, capacity):
        from machine import ADC, Pin, Timer
        self._ADC, self._Pin, self._Timer = ADC, Pin, Timer
        self.ring = Ring(capacity)
        self.channel = None
        self.sample_rate = sample_rate
        self._adc = None
        self._timer = None
        self._set_channel(channel)

    def _set_channel(self, channel):
        if channel not in ADC_PINS:
            raise ValueError('adc_channel must be 0, 1 or 2')
        if channel != self.channel:
            self._adc = self._ADC(self._Pin(ADC_PINS[channel]))
            self.channel = channel

    def _tick(self, _timer):
        self.ring.push(self._adc.read_u16())

    def start(self):
        self.stop()
        self._timer = self._Timer()
        self._timer.init(freq=self.sample_rate, mode=self._Timer.PERIODIC,
                         callback=self._tick)

    def stop(self):
        if self._timer is not None:
            self._timer.deinit()
            self._timer = None

    def reconfigure(self, channel, sample_rate, capacity):
        self.stop()
        self._set_channel(channel)
        self.sample_rate = sample_rate
        if capacity != self.ring.capacity:
            self.ring = Ring(capacity)
        self.start()
