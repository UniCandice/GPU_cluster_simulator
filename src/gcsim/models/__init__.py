"""Physical models.

Each module here answers one question about the hardware, and nothing here knows
what telemetry is. Faults perturb the inputs to these models; telemetry observes
their outputs.

    compute.py   how long does a subdomain take, given cells and device health?
    power.py     how much power does that draw?
    thermal.py   how hot does that get, and what does the GPU do about it?
    network.py   how long does a set of concurrent flows take, and what queues?
    storage.py   how long does a concurrent write take, and what does it cost later?
"""
