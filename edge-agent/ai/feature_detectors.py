"""
Tier-gated feature detectors — footfall, queue length, perimeter breach,
smoke/fire detection. Only active based on client's subscribed tier.
"""
# TODO Phase 3+: Implement per tier matrix


class FootfallCounter:
    """Count persons entering/exiting a zone. Medium+ tier."""
    pass


class PerimeterBreachDetector:
    """Alert when person crosses a defined line/zone. Medium+ tier."""
    pass


class QueueLengthDetector:
    """Estimate queue depth in a defined region. Large+ tier."""
    pass


class SmokFireDetector:
    """Detect smoke/fire events. Large+ tier."""
    pass
