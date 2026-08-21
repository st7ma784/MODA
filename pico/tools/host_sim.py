#!/usr/bin/env python3
"""Run the Pico firmware on a laptop, with a synthetic sensor.

The firmware modules are written so that only `AdcSource` and `netcfg` touch
MicroPython-specific hardware; this script substitutes both, then starts the
real `server.py`. That means the web UI, the config round-trip and the
FastMODA relay can all be developed and debugged against a real FastMODA
instance before any wiring exists.

    python pico/tools/host_sim.py --backend http://localhost:5000

Then open http://localhost:8080/.
"""

import argparse
import asyncio
import math
import os
import random
import sys
import types

FIRMWARE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'firmware')
sys.path.insert(0, os.path.abspath(FIRMWARE))

# `netcfg` imports `network` on use, not on import, but `scan()` would still
# fail here - stub the whole module before `server` can reach it.
_fake_netcfg = types.ModuleType('netcfg')
_fake_netcfg.scan = lambda: [
    {'ssid': 'lab-wifi', 'rssi': -48, 'secure': True},
    {'ssid': 'eduroam', 'rssi': -71, 'secure': True},
    {'ssid': 'pico-guest', 'rssi': -80, 'secure': False},
]
_fake_netcfg.connect = lambda cfg, timeout=20: ('sta', '127.0.0.1')
sys.modules['netcfg'] = _fake_netcfg

import config     # noqa: E402
import sampler    # noqa: E402
import server     # noqa: E402


class SimulatedSource:
    """Fills the same `Ring` the ADC would, from a synthetic waveform.

    Two sinusoids plus noise, in the 0.1-2 Hz band MODA is usually pointed at,
    so a CWT run against it produces something recognisable rather than a flat
    field.
    """

    def __init__(self, sample_rate, capacity, full_scale=3.3):
        self.ring = sampler.Ring(capacity)
        self.sample_rate = sample_rate
        self.full_scale = full_scale
        self._n = 0
        self._task = None

    def _next_count(self):
        t = self._n / self.sample_rate
        volts = (1.65
                 + 0.45 * math.sin(2 * math.pi * 0.25 * t)
                 + 0.20 * math.sin(2 * math.pi * 1.1 * t + 0.7)
                 + 0.02 * random.gauss(0, 1))
        self._n += 1
        counts = int(volts / self.full_scale * sampler.ADC_FULL_SCALE)
        return max(0, min(sampler.ADC_FULL_SCALE, counts))

    async def _run(self):
        # Generate in ticks rather than one-per-sleep: asyncio cannot schedule
        # reliably at 200 Hz, and the shape of the buffer is what matters here.
        tick = 0.05
        while True:
            for _ in range(max(1, int(self.sample_rate * tick))):
                self.ring.push(self._next_count())
            await asyncio.sleep(tick)

    def start(self):
        self.stop()
        self._task = asyncio.get_event_loop().create_task(self._run())

    def stop(self):
        if self._task is not None:
            self._task.cancel()
            self._task = None

    def reconfigure(self, channel, sample_rate, capacity):
        self.stop()
        self.sample_rate = sample_rate
        if capacity != self.ring.capacity:
            self.ring = sampler.Ring(capacity)
        self.start()


async def main(args):
    os.chdir(os.path.abspath(FIRMWARE))     # `server` resolves www/ relatively
    cfg = config.load(args.config)
    if args.backend:
        cfg['backend_url'] = args.backend

    source = SimulatedSource(int(cfg['sample_rate']),
                             config.buffer_samples(cfg),
                             float(cfg['volts_full_scale']))
    source.start()

    harness = server.Harness(cfg, source, 'sta', '127.0.0.1')
    await server.serve(harness, port=args.port)
    print('MODA Pico harness (simulated) on http://localhost:%d/' % args.port)
    print('  backend: %s' % cfg['backend_url'])
    while True:
        await asyncio.sleep(3600)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--backend', help='FastMODA base URL')
    parser.add_argument('--config', default='sim-config.json',
                        help='where the simulator persists settings')
    try:
        asyncio.run(main(parser.parse_args()))
    except KeyboardInterrupt:
        pass
