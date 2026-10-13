#!/usr/bin/env python3
"""Generate deterministic square-grid search missions.

The planner is independent from MAVLink transport.  It converts a selected
centre point and search radius into cells and a lawnmower route that can be
previewed, tested in SITL, and later passed through a guarded command gateway.
"""

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Optional, Tuple


METRES_PER_LATITUDE_DEGREE = 111_320.0


@dataclass(frozen=True)
class SearchCell:
    cell_id: int
    row: int
    column: int
    north_m: float
    east_m: float
    south_m: float
    west_m: float
    center_latitude: float
    center_longitude: float


@dataclass(frozen=True)
class SearchMissionPlan:
    center_latitude: float
    center_longitude: float
    home_latitude: float
    home_longitude: float
    radius_km: float
    altitude_m: float
    cell_count: int
    rows: int
    columns: int
    cells: Tuple[SearchCell, ...]
    route_cell_ids: Tuple[int, ...]
    mission_items: Tuple[Dict[str, Any], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "center": {"latitude": self.center_latitude, "longitude": self.center_longitude},
            "home": {"latitude": self.home_latitude, "longitude": self.home_longitude},
            "radius_km": self.radius_km,
            "altitude_m": self.altitude_m,
            "cell_count": self.cell_count,
            "rows": self.rows,
            "columns": self.columns,
            "cells": [asdict(cell) for cell in self.cells],
            "route_cell_ids": list(self.route_cell_ids),
            "phases": ["takeoff", "grid_search", "return_home", "land"],
            "mission_items": list(self.mission_items),
        }


def build_search_mission(
    center_latitude: float,
    center_longitude: float,
    radius_km: float,
    altitude_m: float,
    home_latitude: Optional[float] = None,
    home_longitude: Optional[float] = None,
) -> SearchMissionPlan:
    """Build a square search divided into four cells per kilometre of radius."""
    center_latitude = _bounded_number("center_latitude", center_latitude, -90.0, 90.0)
    center_longitude = _bounded_number("center_longitude", center_longitude, -180.0, 180.0)
    radius_km = _bounded_number("radius_km", radius_km, 0.25, 10.0)
    altitude_m = _bounded_number("altitude_m", altitude_m, 5.0, 120.0)
    home_latitude = center_latitude if home_latitude is None else _bounded_number(
        "home_latitude", home_latitude, -90.0, 90.0
    )
    home_longitude = center_longitude if home_longitude is None else _bounded_number(
        "home_longitude", home_longitude, -180.0, 180.0
    )

    cell_count = max(1, int(math.ceil(radius_km * 4.0)))
    rows, columns = _closest_factors(cell_count)
    radius_m = radius_km * 1000.0
    cell_height_m = (radius_m * 2.0) / rows
    cell_width_m = (radius_m * 2.0) / columns

    cells: List[SearchCell] = []
    for row in range(rows):
        north_m = radius_m - row * cell_height_m
        south_m = north_m - cell_height_m
        for column in range(columns):
            cell_id = row * columns + column + 1
            east_m = -radius_m + (column + 1) * cell_width_m
            west_m = east_m - cell_width_m
            center_north_m = (north_m + south_m) / 2.0
            center_east_m = (east_m + west_m) / 2.0
            latitude, longitude = _offset_coordinate(
                center_latitude,
                center_longitude,
                center_north_m,
                center_east_m,
            )
            cells.append(SearchCell(
                cell_id=cell_id,
                row=row,
                column=column,
                north_m=round(north_m, 2),
                east_m=round(east_m, 2),
                south_m=round(south_m, 2),
                west_m=round(west_m, 2),
                center_latitude=round(latitude, 7),
                center_longitude=round(longitude, 7),
            ))

    route_cells: List[SearchCell] = []
    for row in range(rows):
        row_cells = [cell for cell in cells if cell.row == row]
        route_cells.extend(row_cells if row % 2 == 0 else reversed(row_cells))

    mission_items: List[Dict[str, Any]] = [
        _mission_item(0, 16, home_latitude, home_longitude, 0.0, current=1),
        _mission_item(1, 22, home_latitude, home_longitude, altitude_m),
    ]
    for cell in route_cells:
        mission_items.append(_mission_item(
            len(mission_items),
            16,
            cell.center_latitude,
            cell.center_longitude,
            altitude_m,
        ))
    mission_items.append(_mission_item(
        len(mission_items),
        16,
        home_latitude,
        home_longitude,
        altitude_m,
    ))
    mission_items.append(_mission_item(
        len(mission_items),
        21,
        home_latitude,
        home_longitude,
        0.0,
    ))

    return SearchMissionPlan(
        center_latitude=center_latitude,
        center_longitude=center_longitude,
        home_latitude=home_latitude,
        home_longitude=home_longitude,
        radius_km=radius_km,
        altitude_m=altitude_m,
        cell_count=cell_count,
        rows=rows,
        columns=columns,
        cells=tuple(cells),
        route_cell_ids=tuple(cell.cell_id for cell in route_cells),
        mission_items=tuple(mission_items),
    )


def _closest_factors(value: int) -> Tuple[int, int]:
    rows = int(math.sqrt(value))
    while rows > 1 and value % rows != 0:
        rows -= 1
    return rows, value // rows


def _offset_coordinate(latitude: float, longitude: float, north_m: float, east_m: float) -> Tuple[float, float]:
    latitude_offset = north_m / METRES_PER_LATITUDE_DEGREE
    longitude_scale = METRES_PER_LATITUDE_DEGREE * max(0.01, math.cos(math.radians(latitude)))
    longitude_offset = east_m / longitude_scale
    return latitude + latitude_offset, longitude + longitude_offset


def _mission_item(
    seq: int,
    command: int,
    latitude: float,
    longitude: float,
    altitude_m: float,
    current: int = 0,
) -> Dict[str, Any]:
    return {
        "seq": seq,
        "frame": 6,
        "command": command,
        "current": current,
        "autocontinue": 1,
        "param1": 0.0,
        "param2": 2.0,
        "param3": 0.0,
        "param4": 0.0,
        "x": int(round(latitude * 1e7)),
        "y": int(round(longitude * 1e7)),
        "z": altitude_m,
    }


def _bounded_number(name: str, value: Any, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(number) or number < minimum or number > maximum:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")
    return number
