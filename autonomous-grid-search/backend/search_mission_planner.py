#!/usr/bin/env python3
"""Generate camera-aware square-grid search missions for simulation review."""

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Optional, Tuple


METRES_PER_LATITUDE_DEGREE = 111_320.0
MAV_CMD_NAV_WAYPOINT = 16
MAV_CMD_NAV_LAND = 21
MAV_CMD_NAV_TAKEOFF = 22
MAV_CMD_DO_SET_CAM_TRIGG_DIST = 206
MAX_MISSION_ITEMS = 500
MAX_SORTIE_MINUTES = 20.0
MAX_SURVEY_LINES = 10_000


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
class SurveyLine:
    cell_id: int
    line_index: int
    start_latitude: float
    start_longitude: float
    end_latitude: float
    end_longitude: float
    length_m: float


@dataclass(frozen=True)
class CameraCoverage:
    horizontal_fov_deg: float
    vertical_fov_deg: float
    side_overlap_percent: float
    forward_overlap_percent: float
    footprint_width_m: float
    footprint_height_m: float
    line_spacing_m: float
    trigger_distance_m: float


@dataclass(frozen=True)
class MissionEstimate:
    survey_line_count: int
    survey_distance_m: float
    total_distance_m: float
    estimated_minutes: float
    estimated_photos: int
    mission_item_count: int
    recommended_sorties: int
    simulation_ready: bool
    blocking_reasons: Tuple[str, ...]
    warnings: Tuple[str, ...]


@dataclass(frozen=True)
class SearchMissionPlan:
    center_latitude: float
    center_longitude: float
    home_latitude: float
    home_longitude: float
    radius_km: float
    altitude_m: float
    cruise_speed_m_s: float
    cell_count: int
    rows: int
    columns: int
    cells: Tuple[SearchCell, ...]
    route_cell_ids: Tuple[int, ...]
    survey_lines: Tuple[SurveyLine, ...]
    camera: CameraCoverage
    estimate: MissionEstimate
    mission_items: Tuple[Dict[str, Any], ...]

    def to_dict(self) -> Dict[str, Any]:
        browser_mission_items = list(self.mission_items) if self.estimate.simulation_ready else []
        return {
            "center": {"latitude": self.center_latitude, "longitude": self.center_longitude},
            "home": {"latitude": self.home_latitude, "longitude": self.home_longitude},
            "radius_km": self.radius_km,
            "altitude_m": self.altitude_m,
            "cruise_speed_m_s": self.cruise_speed_m_s,
            "cell_count": self.cell_count,
            "rows": self.rows,
            "columns": self.columns,
            "cells": [asdict(cell) for cell in self.cells],
            "route_cell_ids": list(self.route_cell_ids),
            "survey_lines": [asdict(line) for line in self.survey_lines],
            "camera": asdict(self.camera),
            "estimate": asdict(self.estimate),
            "surface_tracking": {
                "required": self.altitude_m <= 15.0,
                "source": "downward_rangefinder",
                "terrain_database_is_clearance_source": False,
            },
            "phases": ["takeoff", "camera_grid_survey", "return_home", "land"],
            "mission_items": browser_mission_items,
        }


def build_search_mission(
    center_latitude: float,
    center_longitude: float,
    radius_km: float,
    altitude_m: float,
    home_latitude: Optional[float] = None,
    home_longitude: Optional[float] = None,
    camera_horizontal_fov_deg: float = 78.0,
    camera_vertical_fov_deg: float = 60.0,
    side_overlap_percent: float = 70.0,
    forward_overlap_percent: float = 75.0,
    cruise_speed_m_s: float = 3.0,
) -> SearchMissionPlan:
    """Build a camera survey with parallel flight lines inside every cell."""
    center_latitude = _bounded_number("center_latitude", center_latitude, -85.0, 85.0)
    center_longitude = _bounded_number("center_longitude", center_longitude, -180.0, 180.0)
    radius_km = _bounded_number("radius_km", radius_km, 0.01, 10.0)
    altitude_m = _bounded_number("altitude_m", altitude_m, 5.0, 120.0)
    home_latitude = center_latitude if home_latitude is None else _bounded_number(
        "home_latitude", home_latitude, -85.0, 85.0
    )
    home_longitude = center_longitude if home_longitude is None else _bounded_number(
        "home_longitude", home_longitude, -180.0, 180.0
    )
    horizontal_fov = _bounded_number("camera_horizontal_fov_deg", camera_horizontal_fov_deg, 10.0, 170.0)
    vertical_fov = _bounded_number("camera_vertical_fov_deg", camera_vertical_fov_deg, 10.0, 170.0)
    side_overlap = _bounded_number("side_overlap_percent", side_overlap_percent, 0.0, 95.0)
    forward_overlap = _bounded_number("forward_overlap_percent", forward_overlap_percent, 0.0, 95.0)
    cruise_speed = _bounded_number("cruise_speed_m_s", cruise_speed_m_s, 0.5, 15.0)

    footprint_width = 2.0 * altitude_m * math.tan(math.radians(horizontal_fov / 2.0))
    footprint_height = 2.0 * altitude_m * math.tan(math.radians(vertical_fov / 2.0))
    line_spacing = footprint_height * (1.0 - side_overlap / 100.0)
    trigger_distance = footprint_width * (1.0 - forward_overlap / 100.0)
    if line_spacing < 0.25 or trigger_distance < 0.25:
        raise ValueError("camera overlap creates spacing below the 0.25 m planning limit")

    camera = CameraCoverage(
        horizontal_fov_deg=horizontal_fov,
        vertical_fov_deg=vertical_fov,
        side_overlap_percent=side_overlap,
        forward_overlap_percent=forward_overlap,
        footprint_width_m=round(footprint_width, 2),
        footprint_height_m=round(footprint_height, 2),
        line_spacing_m=round(line_spacing, 2),
        trigger_distance_m=round(trigger_distance, 2),
    )

    cell_count = max(1, int(math.ceil(radius_km * 4.0)))
    rows, columns = _closest_factors(cell_count)
    radius_m = radius_km * 1000.0
    cell_height_m = (radius_m * 2.0) / rows
    cell_width_m = (radius_m * 2.0) / columns
    cell_line_count = max(1, int(math.ceil(cell_height_m / line_spacing)))
    predicted_line_count = cell_count * cell_line_count
    if predicted_line_count > MAX_SURVEY_LINES:
        raise ValueError(
            f"survey requires {predicted_line_count} flight lines; reduce the area or overlap, "
            f"or increase altitude (planning limit {MAX_SURVEY_LINES} lines)"
        )

    cells: List[SearchCell] = []
    for row in range(rows):
        north_m = radius_m - row * cell_height_m
        south_m = north_m - cell_height_m
        for column in range(columns):
            cell_id = row * columns + column + 1
            east_m = -radius_m + (column + 1) * cell_width_m
            west_m = east_m - cell_width_m
            latitude, longitude = _offset_coordinate(
                center_latitude,
                center_longitude,
                (north_m + south_m) / 2.0,
                (east_m + west_m) / 2.0,
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

    survey_lines: List[SurveyLine] = []
    route_offsets: List[Tuple[float, float]] = []
    survey_distance_m = 0.0
    for cell_order, cell in enumerate(route_cells):
        actual_spacing = cell_height_m / cell_line_count
        for line_index in range(cell_line_count):
            north_m = cell.north_m - actual_spacing * (line_index + 0.5)
            forward = (cell_order + line_index) % 2 == 0
            start_east_m = cell.west_m if forward else cell.east_m
            end_east_m = cell.east_m if forward else cell.west_m
            start_latitude, start_longitude = _offset_coordinate(
                center_latitude, center_longitude, north_m, start_east_m
            )
            end_latitude, end_longitude = _offset_coordinate(
                center_latitude, center_longitude, north_m, end_east_m
            )
            survey_lines.append(SurveyLine(
                cell_id=cell.cell_id,
                line_index=line_index + 1,
                start_latitude=round(start_latitude, 7),
                start_longitude=round(start_longitude, 7),
                end_latitude=round(end_latitude, 7),
                end_longitude=round(end_longitude, 7),
                length_m=round(cell_width_m, 2),
            ))
            route_offsets.extend([(north_m, start_east_m), (north_m, end_east_m)])
            survey_distance_m += cell_width_m

    home_offset = _coordinate_offset_metres(
        center_latitude, center_longitude, home_latitude, home_longitude
    )
    total_distance_m = _polyline_distance([home_offset] + route_offsets + [home_offset])
    mission_item_count = 6 + len(route_offsets)
    estimated_minutes = (total_distance_m / cruise_speed + 120.0) / 60.0
    estimated_photos = int(math.ceil(survey_distance_m / trigger_distance))
    recommended_sorties = max(
        1,
        int(math.ceil(mission_item_count / MAX_MISSION_ITEMS)),
        int(math.ceil(estimated_minutes / MAX_SORTIE_MINUTES)),
    )

    blocking_reasons: List[str] = []
    if mission_item_count > MAX_MISSION_ITEMS:
        blocking_reasons.append(
            f"mission has {mission_item_count} items; split it below {MAX_MISSION_ITEMS} items per sortie"
        )
    if estimated_minutes > MAX_SORTIE_MINUTES:
        blocking_reasons.append(
            f"estimated flight time is {estimated_minutes:.1f} minutes; split it below {MAX_SORTIE_MINUTES:.0f} minutes per sortie"
        )
    warnings: List[str] = []
    if altitude_m <= 15.0:
        warnings.append("low-altitude flight requires a healthy downward rangefinder; terrain elevation is not a clearance sensor")
    if radius_km >= 1.0 and altitude_m <= 10.0:
        warnings.append("this area is extremely large for low-altitude camera coverage")

    estimate = MissionEstimate(
        survey_line_count=len(survey_lines),
        survey_distance_m=round(survey_distance_m, 1),
        total_distance_m=round(total_distance_m, 1),
        estimated_minutes=round(estimated_minutes, 1),
        estimated_photos=estimated_photos,
        mission_item_count=mission_item_count,
        recommended_sorties=recommended_sorties,
        simulation_ready=not blocking_reasons,
        blocking_reasons=tuple(blocking_reasons),
        warnings=tuple(warnings),
    )

    mission_items: List[Dict[str, Any]] = [
        _navigation_item(0, MAV_CMD_NAV_WAYPOINT, home_latitude, home_longitude, 0.0, current=1),
        _navigation_item(1, MAV_CMD_NAV_TAKEOFF, home_latitude, home_longitude, altitude_m),
        _camera_trigger_item(2, trigger_distance),
    ]
    for line in survey_lines:
        mission_items.append(_navigation_item(
            len(mission_items), MAV_CMD_NAV_WAYPOINT,
            line.start_latitude, line.start_longitude, altitude_m,
        ))
        mission_items.append(_navigation_item(
            len(mission_items), MAV_CMD_NAV_WAYPOINT,
            line.end_latitude, line.end_longitude, altitude_m,
        ))
    mission_items.append(_camera_trigger_item(len(mission_items), 0.0))
    mission_items.append(_navigation_item(
        len(mission_items), MAV_CMD_NAV_WAYPOINT,
        home_latitude, home_longitude, altitude_m,
    ))
    mission_items.append(_navigation_item(
        len(mission_items), MAV_CMD_NAV_LAND,
        home_latitude, home_longitude, 0.0,
    ))

    return SearchMissionPlan(
        center_latitude=center_latitude,
        center_longitude=center_longitude,
        home_latitude=home_latitude,
        home_longitude=home_longitude,
        radius_km=radius_km,
        altitude_m=altitude_m,
        cruise_speed_m_s=cruise_speed,
        cell_count=cell_count,
        rows=rows,
        columns=columns,
        cells=tuple(cells),
        route_cell_ids=tuple(cell.cell_id for cell in route_cells),
        survey_lines=tuple(survey_lines),
        camera=camera,
        estimate=estimate,
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
    return latitude + latitude_offset, longitude + east_m / longitude_scale


def _coordinate_offset_metres(
    origin_latitude: float,
    origin_longitude: float,
    latitude: float,
    longitude: float,
) -> Tuple[float, float]:
    north_m = (latitude - origin_latitude) * METRES_PER_LATITUDE_DEGREE
    longitude_scale = METRES_PER_LATITUDE_DEGREE * max(0.01, math.cos(math.radians(origin_latitude)))
    east_m = (longitude - origin_longitude) * longitude_scale
    return north_m, east_m


def _polyline_distance(points: List[Tuple[float, float]]) -> float:
    return sum(
        math.hypot(current[0] - previous[0], current[1] - previous[1])
        for previous, current in zip(points, points[1:])
    )


def _navigation_item(
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


def _camera_trigger_item(seq: int, trigger_distance_m: float) -> Dict[str, Any]:
    return {
        "seq": seq,
        "frame": 2,
        "command": MAV_CMD_DO_SET_CAM_TRIGG_DIST,
        "current": 0,
        "autocontinue": 1,
        "param1": round(trigger_distance_m, 2),
        "param2": 0.0,
        "param3": 1.0 if trigger_distance_m > 0 else 0.0,
        "param4": 0.0,
        "x": 0,
        "y": 0,
        "z": 0.0,
    }


def _bounded_number(name: str, value: Any, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(number) or number < minimum or number > maximum:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")
    return number
