"""Wi-Fi bring-up for the Pico 2 W, with an access-point fallback.

If the configured network is unreachable (wrong password, out of range, first
boot with no credentials) the board raises its own AP instead of sitting
headless: connect to it and the same web UI is there to fix the settings.
"""

import time

AP_ADDRESS = '192.168.4.1'


def connect(cfg, timeout=20):
    """Join the configured network. Returns `(mode, ip)`; mode is 'sta' or 'ap'."""
    import network

    ssid = cfg.get('wifi_ssid') or ''
    if ssid:
        station = network.WLAN(network.STA_IF)
        station.active(True)
        try:
            station.config(hostname=cfg.get('hostname') or 'moda-pico')
        except (OSError, ValueError):
            pass  # older firmware without hostname support
        station.connect(ssid, cfg.get('wifi_password') or '')
        deadline = time.time() + timeout
        while time.time() < deadline:
            if station.isconnected():
                return 'sta', station.ifconfig()[0]
            time.sleep(0.5)
        station.active(False)

    return 'ap', start_ap(cfg)


def start_ap(cfg):
    import network

    access_point = network.WLAN(network.AP_IF)
    access_point.config(essid=cfg.get('ap_ssid') or 'moda-pico',
                        password=cfg.get('ap_password') or 'modamoda')
    access_point.active(True)
    while not access_point.active():
        time.sleep(0.2)
    return access_point.ifconfig()[0]


def scan():
    """Visible networks, strongest first: `[{'ssid', 'rssi', 'secure'}, ...]`."""
    import network

    station = network.WLAN(network.STA_IF)
    was_active = station.active()
    station.active(True)
    try:
        found = {}
        for entry in station.scan():
            ssid = entry[0].decode('utf-8', 'replace').strip()
            if not ssid:
                continue  # hidden network - nothing useful to show
            rssi, security = entry[3], entry[4]
            if ssid not in found or rssi > found[ssid]['rssi']:
                found[ssid] = {'ssid': ssid, 'rssi': rssi, 'secure': security > 0}
        return sorted(found.values(), key=lambda item: item['rssi'], reverse=True)
    finally:
        if not was_active:
            station.active(False)
