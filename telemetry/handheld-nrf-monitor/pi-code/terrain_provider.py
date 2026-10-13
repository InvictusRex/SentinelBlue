#!/usr/bin/env python3
"""Optional terrain elevation sampling for mission planning context."""

import json
import os
from typing import Any, Dict, Iterable, Tuple
from urllib.request import Request, urlopen


DEFAULT_TERRAIN_API_URL = "https://api.opentopodata.org/v1/srtm90m"


def sample_plan_terrain(plan, surface_type: str = "water") -> Dict[str, Any]:
    """Sample home, centre, and cell centres without using DEM as clearance data."""
    points = [
        (plan.home_latitude, plan.home_longitude),
        (plan.center_latitude, plan.center_longitude),
    ]
    points.extend((cell.center_latitude, cell.center_longitude) for cell in plan.cells)
    result = sample_elevations(points)
    result["surface_type"] = surface_type
    result["used_for_clearance"] = False
    result["clearance_source"] = "downward_rangefinder" if surface_type == "water" else "rangefinder_or_validated_terrain"
    return result


def sample_elevations(points: Iterable[Tuple[float, float]]) -> Dict[str, Any]:
    point_list = list(points)
    api_url = os.environ.get("TERRAIN_API_URL", DEFAULT_TERRAIN_API_URL).rstrip("/")
    locations = "|".join(f"{latitude:.7f},{longitude:.7f}" for latitude, longitude in point_list)
    request = Request(
        api_url,
        data=json.dumps({"locations": locations, "interpolation": "cubic"}).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "SentinelBlue-Companion/1.0"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=5) as response:
            payload = json.load(response)
        elevations = [
            float(item["elevation"])
            for item in payload.get("results", [])
            if item.get("elevation") is not None
        ]
        if not elevations:
            return {
                "status": "unavailable",
                "provider": api_url,
                "sample_count": 0,
                "message": "terrain service returned no elevations for this area",
            }
        return {
            "status": "available",
            "provider": api_url,
            "sample_count": len(elevations),
            "minimum_elevation_m": round(min(elevations), 1),
            "maximum_elevation_m": round(max(elevations), 1),
            "elevation_range_m": round(max(elevations) - min(elevations), 1),
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "provider": api_url,
            "sample_count": 0,
            "message": f"terrain service unavailable: {type(exc).__name__}",
        }
