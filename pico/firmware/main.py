"""Entry point on the Pico 2 W. Runs automatically at boot.

Order matters: config, then network, then the sampler, then the server. If the
network step falls back to AP mode the rest still comes up, so the UI is always
reachable to fix whatever was wrong.
"""

try:
    import asyncio
except ImportError:
    import uasyncio as asyncio

import config
import netcfg
import sampler
import server


async def _heartbeat():
    """Onboard LED: slow blink = running. The only status the board can show
    before you have found its IP address."""
    try:
        from machine import Pin
        led = Pin('LED', Pin.OUT)
    except (ImportError, ValueError):
        return
    while True:
        led.toggle()
        await asyncio.sleep(1)


async def _main():
    cfg = config.load()

    mode, ip = netcfg.connect(cfg)
    if mode == 'ap':
        print('Wi-Fi join failed or unconfigured; serving AP "%s"'
              % cfg['ap_ssid'])
    print('MODA Pico harness: http://%s/  (mode=%s)' % (ip, mode))

    source = sampler.AdcSource(int(cfg['adc_channel']),
                               int(cfg['sample_rate']),
                               config.buffer_samples(cfg))
    source.start()

    harness = server.Harness(cfg, source, mode, ip)
    await server.serve(harness)
    await _heartbeat()


try:
    asyncio.run(_main())
finally:
    asyncio.new_event_loop()
