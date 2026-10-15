"""Polygon camera surveys in a local metric projection; review/simulation only.

Tracks are clipped to the survey polygon. Transit legs and optional run-outs
are displayed explicitly and are NOT obstacle/geofence avoidance routes.
"""

from dataclasses import dataclass
import math
import json
import pathlib
import time
from typing import Any, Dict

from pyproj import CRS, Transformer, Geod
from shapely import affinity
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union
from shapely.validation import explain_validity

from search_mission_planner import _bounded_number, _navigation_item, _camera_trigger_item

MAX_LINES = 10000
MAX_ITEMS = 500
MAX_SORTIES = 100


def build_radius_survey(payload, telemetry):
    """The browser supplies an area only; all flight settings belong to the Pi."""
    area = payload.get("search_area")
    if not isinstance(area, dict):
        raise ValueError("search_area must contain center and radius_m")
    center = position(area.get("center"), "search center")
    radius = number(area, "radius_m", None, 2, 10000)
    if telemetry.get("armed"):
        raise ValueError("Finish the current flight before planning from a new launch position")
    try:
        config = json.loads(pathlib.Path(__file__).with_name("survey_config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Pi survey_config.json is missing or invalid; check the aircraft configuration") from exc
    if not isinstance(config, dict) or not isinstance(config.get("survey_options"), dict):
        raise ValueError("Pi survey_config.json must include survey_options")
    geod = Geod(ellps="WGS84")
    boundary = []
    for index in range(64):
        # Circumscribe the requested circle so polygon approximation never
        # silently excludes the rim of the user's search area.
        longitude, latitude, _ = geod.fwd(center["longitude"], center["latitude"], index * 360 / 64,
                                         radius / math.cos(math.pi / 64))
        boundary.append({"latitude": latitude, "longitude": longitude})
    source = dict(telemetry)
    try:
        fresh = all(0 <= time.time() - float(source.get(key, 0)) <= 5
                    for key in ("last_position_timestamp", "last_gps_timestamp"))
    except (TypeError, ValueError):
        fresh = False
    if not fresh or source.get("gps_fix_type") not in ("3D FIX", "DGPS", "RTK FLT", "RTK FIX"):
        source["connected"] = False
    request = {**config, "boundary": boundary}
    request.pop("home", None)
    plan = build_auto_grid_mission(request, source)
    plan.data["search_area"] = {"center": center, "radius_m": radius}
    plan.data["profile_name"] = str(config.get("profile_name", "pi_survey_profile"))
    plan.data["terrain_enabled"] = config.get("terrain_enabled") is True
    plan.data["surface_type"] = config.get("surface_type", "water")
    if plan.data["home_source"] == "area_preview_only":
        reason = "Pi needs fresh 3D GPS from the disarmed vehicle to determine launch/home"
        plan.data["estimate"]["blocking_reasons"] = [reason] + plan.data["estimate"]["blocking_reasons"][1:]
        for sortie in plan.data["sorties"]:
            sortie["blocking_reasons"][0] = reason
    return plan


def number(data, key, default, minimum, maximum):
    value = data.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a number")
    return _bounded_number(key, value, minimum, maximum)


def choice(data, key, default, options):
    value = data.get(key, default)
    if value not in options:
        raise ValueError(f"{key} must be one of {', '.join(options)}")
    return value


def position(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain latitude and longitude")
    return {
        "latitude": number(value, "latitude", None, -85, 85),
        "longitude": number(value, "longitude", None, -180, 180),
    }


@dataclass
class AutoGridPlan:
    data: Dict[str, Any]

    @property
    def altitude_m(self):
        return self.data["altitude_m"]

    @property
    def home_latitude(self):
        return self.data["home"]["latitude"]

    @property
    def home_longitude(self):
        return self.data["home"]["longitude"]

    @property
    def center_latitude(self):
        return self.data["center"]["latitude"]

    @property
    def center_longitude(self):
        return self.data["center"]["longitude"]

    @property
    def cells(self):
        return ()

    def to_dict(self):
        return self.data


def build_auto_grid_mission(payload, telemetry):
    options = payload.get("survey_options")
    if not isinstance(options, dict):
        raise ValueError("survey_options must be an object")
    raw_boundary = payload.get("boundary")
    if not isinstance(raw_boundary, list) or not 3 <= len(raw_boundary) <= 64:
        raise ValueError("Draw a closed boundary with 3 to 64 vertices")
    boundary = [position(point, "boundary vertex") for point in raw_boundary]
    if boundary[0] == boundary[-1]:
        boundary.pop()
    if len({(p['latitude'], p['longitude']) for p in boundary}) != len(boundary):
        raise ValueError("Boundary contains duplicate vertices; remove repeated points")
    if len(boundary) < 3:
        raise ValueError("Boundary needs at least 3 distinct vertices")
    lat = sum(p["latitude"] for p in boundary) / len(boundary)
    lon = sum(p["longitude"] for p in boundary) / len(boundary)
    local_crs = CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m")
    to_local = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)
    to_geo = Transformer.from_crs(local_crs, "EPSG:4326", always_xy=True)
    polygon = Polygon([to_local.transform(p["longitude"], p["latitude"]) for p in boundary])
    if not polygon.is_valid:
        raise ValueError(f"Invalid boundary: {explain_validity(polygon)}. Redraw without crossing edges")
    if polygon.area < 4:
        raise ValueError("Survey area must be at least 4 square metres")
    x0, y0, x1, y1 = polygon.bounds
    if max(x1 - x0, y1 - y0) > 20000:
        raise ValueError("Survey boundary must fit within a 20 km extent; divide the area")

    def geo(point):
        longitude, latitude = to_geo.transform(*point)
        return {"latitude": round(latitude, 7), "longitude": round(longitude, 7)}

    home_source = "manual"
    if payload.get("home") is not None:
        home = position(payload["home"], "home")
    elif telemetry.get("connected") and telemetry.get("latitude") is not None and telemetry.get("longitude") is not None:
        home = position(telemetry, "telemetry position")
        home_source = "vehicle_position"
    else:
        home = geo(polygon.representative_point().coords[0])
        home_source = "area_preview_only"
    home_xy = to_local.transform(home["longitude"], home["latitude"])
    if math.hypot(*home_xy) > 30000:
        raise ValueError("Home is more than 30 km from the area; set a nearby launch position")

    altitude = number(payload, "altitude_m", 5, 5, 120)
    speed = number(payload, "cruise_speed_m_s", 2, 0.5, 15)
    side = number(payload, "side_overlap_percent", 70, 0, 95)
    forward = number(payload, "forward_overlap_percent", 75, 0, 95)
    model = choice(options, "camera_model", "fov", ("fov", "sensor"))
    if model == "sensor":
        sensor_width = number(options, "sensor_width_mm", 6.17, 1, 60)
        sensor_height = number(options, "sensor_height_mm", 4.55, 1, 60)
        focal = number(options, "focal_length_mm", 4.5, 1, 300)
        hfov = math.degrees(2 * math.atan(sensor_width / (2 * focal)))
        vfov = math.degrees(2 * math.atan(sensor_height / (2 * focal)))
    else:
        hfov = number(payload, "camera_horizontal_fov_deg", 78, 10, 150)
        vfov = number(payload, "camera_vertical_fov_deg", 60, 10, 150)
    width = 2 * altitude * math.tan(math.radians(hfov / 2))
    height = 2 * altitude * math.tan(math.radians(vfov / 2))
    image_width = number(options, "image_width_px", 4000, 100, 30000)
    image_height = number(options, "image_height_px", 3000, 100, 30000)
    orientation = choice(options, "camera_orientation", "landscape", ("landscape", "portrait"))
    across, along = (width, height) if orientation == "landscape" else (height, width)
    automatic_spacing = across * (1 - side / 100)
    trigger = round(along * (1 - forward / 100), 6)
    spacing_mode = choice(options, "spacing_mode", "camera", ("camera", "manual"))
    spacing = automatic_spacing if spacing_mode == "camera" else number(options, "line_spacing_m", 2, 0.25, 1000)
    if min(spacing, trigger) < 0.25:
        raise ValueError("Spacing below 0.25 m is unsupported; adjust altitude or overlap")
    min_interval = number(options, "minimum_trigger_interval_s", 0.5, 0.1, 60)
    overshoot = number(options, "overshoot_m", 0, 0, 50)
    endurance = number(options, "endurance_minutes", 20, 3, 120)
    reserve = number(options, "reserve_percent", 20, 10, 50)
    budget = endurance * (1 - reserve / 100)
    turn_seconds = 2 + number(options, "turn_seconds", 3, 0, 30)
    angle_mode = choice(options, "angle_mode", "auto", ("auto", "manual"))
    requested_angle = number(options, "grid_angle_deg", 90, 0, 179.99)
    entry = choice(options, "entry", "nearest", ("nearest", "reverse"))
    pattern = choice(options, "pattern", "single", ("single", "crosshatch"))

    def make_tracks(heading):
        # Rotated x is along flight; y is across flight. Bearing is clockwise from north.
        rotation = heading - 90
        rotated = affinity.rotate(polygon, rotation, origin=(0, 0))
        left, bottom, right, top = rotated.bounds
        count = max(1, math.ceil((top - bottom) / spacing))
        if count > MAX_LINES:
            raise ValueError("Grid exceeds 10000 lines; reduce the area or increase spacing")
        actual = (top - bottom) / count
        tracks = []
        for row in range(count):
            y = bottom + (row + 0.5) * actual
            cut = rotated.intersection(LineString([(left - 1, y), (right + 1, y)]))
            pieces = [cut] if cut.geom_type == "LineString" else list(getattr(cut, "geoms", []))
            pieces = sorted((p for p in pieces if p.geom_type == "LineString" and p.length > 0.01), key=lambda p: p.bounds[0])
            if row % 2:
                pieces.reverse()
            for piece in pieces:
                start, end = (piece.bounds[0] - overshoot, y), (piece.bounds[2] + overshoot, y)
                if row % 2:
                    start, end = end, start
                line = affinity.rotate(LineString([start, end]), -rotation, origin=(0, 0))
                tracks.append((tuple(line.coords[0]), tuple(line.coords[-1]), heading))
        if len(tracks) > MAX_LINES:
            raise ValueError("Grid exceeds 10000 line segments; divide the boundary")
        return tracks, actual

    def duration(tracks):
        distance = 0.0
        previous = home_xy
        for start, end, _ in tracks:
            distance += math.dist(previous, start) + math.dist(start, end)
            previous = end
        distance += math.dist(previous, home_xy)
        return distance, (distance / speed + 120 + turn_seconds * len(tracks)) / 60

    if angle_mode == "auto":
        # Bounded candidate search, not a claim of a globally optimal route.
        candidates = list(range(0, 180, 15))
        edge = max(zip(polygon.exterior.coords, list(polygon.exterior.coords)[1:]), key=lambda p: math.dist(*p))
        candidates.append(math.degrees(math.atan2(edge[1][0] - edge[0][0], edge[1][1] - edge[0][1])) % 180)
        choices = []
        for candidate in candidates:
            try:
                tracks, _ = make_tracks(candidate)
                if tracks:
                    choices.append((duration(tracks)[1], candidate))
            except ValueError:
                continue
        if not choices:
            raise ValueError("No grid within the 10000 line limit; divide the survey area")
        angle = min(choices)[1]
    else:
        angle = requested_angle
    tracks, actual_spacing = make_tracks(angle)
    if pattern == "crosshatch":
        second, _ = make_tracks((angle + 90) % 180)
        tracks += second
    if not tracks or len(tracks) > MAX_LINES:
        raise ValueError("Grid must contain between 1 and 10000 line segments")
    if math.dist(home_xy, tracks[-1][1]) < math.dist(home_xy, tracks[0][0]):
        tracks = [(end, start, heading) for start, end, heading in reversed(tracks)]
    if entry == "reverse":
        tracks = [(end, start, heading) for start, end, heading in reversed(tracks)]

    footprints = []
    for start, end, heading in tracks:
        rotation = heading - 90
        line = affinity.rotate(LineString([start, end]), rotation, origin=(0, 0))
        left, y, right, _ = line.bounds
        # Continuous footprint union, conservative for discrete triggering when
        # the final photograph is up to one trigger distance short of the end.
        if line.coords[0][0] <= line.coords[-1][0]:
            right = left + math.floor((right - left) / trigger) * trigger
        else:
            left = right - math.floor((right - left) / trigger) * trigger
        footprints.append(affinity.rotate(box(left - along / 2, y - across / 2, right + along / 2, y + across / 2), -rotation, origin=(0, 0)))
    covered = unary_union(footprints)
    coverage = min(100, covered.intersection(polygon).area / polygon.area * 100)
    blockers = []
    warnings = ["Flat-surface, nadir-camera model; terrain samples do not change mission altitudes.",
                "Transit legs are straight connections, not obstacle or geofence avoidance.",
                "ArduCopter setup: validate face-next-waypoint yaw, waypoint radius smaller than the line gap, and the camera trigger interface in SITL. Waypoint yaw/radius fields do not configure these settings."]
    if home_source == "area_preview_only":
        blockers.append("Set a launch/home position before export or simulation")
    if coverage < 99.5:
        blockers.append(f"Predicted footprint covers {coverage:.2f}% of the boundary; increase overlap or change grid angle")
    if spacing > automatic_spacing + 0.001:
        blockers.append("Manual line spacing exceeds the requested side-overlap spacing")
    if trigger / speed < min_interval - 1e-9:
        blockers.append(f"Camera cannot keep up: reduce speed to {trigger / min_interval:.2f} m/s or change forward overlap")
    if altitude <= 15:
        warnings.append("Low-altitude water clearance requires a validated sensor and autopilot setup; this planner does not configure surface following.")

    groups = []
    group = []
    group_distance = 0.0
    for track in tracks:
        start, end, _ = track
        previous = group[-1][1] if group else home_xy
        candidate_distance = group_distance + math.dist(previous, start) + math.dist(start, end)
        candidate_minutes = ((candidate_distance + math.dist(end, home_xy)) / speed + 120 + turn_seconds * (len(group) + 1)) / 60
        if group and (5 + 4 * (len(group) + 1) > MAX_ITEMS or candidate_minutes > budget):
            groups.append(group)
            group = []
            candidate_distance = math.dist(home_xy, start) + math.dist(start, end)
        group.append(track)
        group_distance = candidate_distance
    if group:
        groups.append(group)
    if len(groups) > MAX_SORTIES:
        raise ValueError("Survey needs more than 100 sorties; divide it into smaller areas")

    sorties, survey_lines = [], []
    # A centimetre tolerance avoids classifying coincident boundary transfers
    # as outside because of floating-point intersection coordinates.
    boundary_tolerance = polygon.buffer(0.01)
    for group_index, group in enumerate(groups):
        distance, minutes = duration(group)
        items = [_navigation_item(0, 16, home["latitude"], home["longitude"], 0, current=1),
                 _navigation_item(1, 22, home["latitude"], home["longitude"], altitude)]
        speed_item = _camera_trigger_item(2, 0)
        speed_item.update(command=178, param1=1, param2=speed, param3=-1)
        items.append(speed_item)
        photo_count = 0
        path = [home]
        outside = 0.0
        previous = home_xy
        for start, end, heading in group:
            start_geo, end_geo = geo(start), geo(end)
            length = math.dist(start, end)
            for point, before in ((start_geo, True), (end_geo, False)):
                item = _navigation_item(len(items), 16, point["latitude"], point["longitude"], altitude)
                # Copter uses its configured yaw behaviour and waypoint radius;
                # it does not implement per-waypoint yaw/acceptance parameters.
                item.update(param1=1.0, param2=0.0, param4=0.0)
                items.append(item)
                trigger_item = _camera_trigger_item(len(items), trigger if before else 0)
                trigger_item["param1"] = trigger if before else 0
                items.append(trigger_item)
            photo_count += math.floor(length / trigger) + 1
            outside += LineString([previous, start]).difference(boundary_tolerance).length
            outside += LineString([start, end]).difference(boundary_tolerance).length
            path.extend([start_geo, end_geo])
            survey_lines.append({"cell_id": 1, "line_index": len(survey_lines) + 1,
                                 "sortie_id": group_index + 1, "start_latitude": start_geo["latitude"],
                                 "start_longitude": start_geo["longitude"], "end_latitude": end_geo["latitude"],
                                 "end_longitude": end_geo["longitude"], "length_m": round(length, 2)})
            previous = end
        outside += LineString([previous, home_xy]).difference(boundary_tolerance).length
        path.append(home)
        items.append(_navigation_item(len(items), 16, home["latitude"], home["longitude"], altitude))
        items.append(_navigation_item(len(items), 21, home["latitude"], home["longitude"], 0))
        local_blockers = list(blockers)
        if minutes > budget:
            local_blockers.append("A single line plus home transit exceeds the sortie budget; reduce the area or move home")
        sorties.append({"sortie_id": group_index + 1, "line_count": len(group), "route": path,
                        "distance_m": round(distance, 1), "estimated_minutes": round(minutes, 1),
                        "estimated_photos": photo_count, "outside_boundary_m": round(outside, 1),
                        "mission_item_count": len(items), "ready": not local_blockers,
                        "blocking_reasons": local_blockers, "mission_items": items if not local_blockers else []})

    camera = {"horizontal_fov_deg": round(hfov, 2), "vertical_fov_deg": round(vfov, 2),
              "footprint_width_m": round(width, 3), "footprint_height_m": round(height, 3),
              "cross_track_m": round(across, 3), "along_track_m": round(along, 3),
              "line_spacing_m": round(actual_spacing, 3), "requested_line_spacing_m": round(spacing, 3),
              "trigger_distance_m": round(trigger, 3), "trigger_interval_s": round(trigger / speed, 3),
              "effective_side_overlap_percent": round(100 * (1 - actual_spacing / across), 2),
              "gsd_width_cm": round(width / image_width * 100, 3),
              "gsd_height_cm": round(height / image_height * 100, 3)}
    return AutoGridPlan({"schema_version": 2, "center": geo(polygon.centroid.coords[0]), "home": home,
                        "home_source": home_source, "boundary": boundary, "altitude_m": altitude,
                        "altitude_reference": "relative_home", "camera": camera,
                        "cruise_speed_m_s": speed, "survey_lines": survey_lines, "sorties": sorties,
                        "grid_angle_deg": round(angle, 2), "pattern": pattern, "area_m2": round(polygon.area, 1),
                        "coverage_percent": round(coverage, 2), "sortie_budget_minutes": round(budget, 2),
                        "estimate": {"survey_line_count": len(tracks), "recommended_sorties": len(sorties),
                                     "total_distance_m": round(sum(s["distance_m"] for s in sorties), 1),
                                     "estimated_minutes": round(sum(s["estimated_minutes"] for s in sorties), 1),
                                     "estimated_photos": sum(s["estimated_photos"] for s in sorties),
                                     "simulation_ready": all(s["ready"] for s in sorties),
                                     "blocking_reasons": list(dict.fromkeys(r for s in sorties for r in s["blocking_reasons"])),
                                     "warnings": warnings}})
