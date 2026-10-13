import { useEffect, useRef } from "react";
import {
  LngLatBounds,
  Map,
  NavigationControl,
  setWorkerUrl,
  type GeoJSONSource,
  type MapMouseEvent,
  type StyleSpecification
} from "maplibre-gl";
import mapLibreWorkerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?url";

setWorkerUrl(mapLibreWorkerUrl);

interface MapCell {
  cell_id: number;
  north_m: number;
  east_m: number;
  south_m: number;
  west_m: number;
}

interface MapSurveyLine {
  cell_id: number;
  start_latitude: number;
  start_longitude: number;
  end_latitude: number;
  end_longitude: number;
}

interface MapPlan {
  center: { latitude: number; longitude: number };
  home: { latitude: number; longitude: number };
  radius_km: number;
  cells: MapCell[];
  survey_lines: MapSurveyLine[];
}

const emptyCollection: GeoJSON.FeatureCollection = { type: "FeatureCollection", features: [] };

function mapStyle(): StyleSpecification | string {
  const mapTilerKey = (import.meta as ImportMeta & {
    env?: Record<string, string | undefined>;
  }).env?.VITE_MAPTILER_KEY;
  if (mapTilerKey) {
    return `https://api.maptiler.com/maps/hybrid/style.json?key=${encodeURIComponent(mapTilerKey)}`;
  }
  return {
    version: 8,
    sources: {
      open_street_map: {
        type: "raster",
        tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
        tileSize: 256,
        attribution: "© OpenStreetMap contributors"
      },
      terrain_dem: {
        type: "raster-dem",
        url: "https://demotiles.maplibre.org/terrain-tiles/tiles.json",
        tileSize: 256
      },
      terrain_hillshade_dem: {
        type: "raster-dem",
        url: "https://demotiles.maplibre.org/terrain-tiles/tiles.json",
        tileSize: 256
      }
    },
    layers: [
      { id: "open_street_map", type: "raster", source: "open_street_map" },
      {
        id: "terrain_hillshade",
        type: "hillshade",
        source: "terrain_hillshade_dem",
        paint: {
          "hillshade-exaggeration": 0.35,
          "hillshade-shadow-color": "#071014",
          "hillshade-highlight-color": "#d7fff8",
          "hillshade-accent-color": "#0f766e"
        }
      }
    ]
  };
}

function offsetCoordinate(latitude: number, longitude: number, northM: number, eastM: number): [number, number] {
  const nextLatitude = latitude + northM / 111_320;
  const longitudeScale = 111_320 * Math.max(0.01, Math.cos(latitude * Math.PI / 180));
  return [longitude + eastM / longitudeScale, nextLatitude];
}

function planGeometry(plan: MapPlan | null, selected: { latitude: number; longitude: number } | null) {
  const areas: GeoJSON.Feature[] = [];
  const lines: GeoJSON.Feature[] = [];
  const points: GeoJSON.Feature[] = [];

  if (plan) {
    for (const cell of plan.cells) {
      const northWest = offsetCoordinate(plan.center.latitude, plan.center.longitude, cell.north_m, cell.west_m);
      const northEast = offsetCoordinate(plan.center.latitude, plan.center.longitude, cell.north_m, cell.east_m);
      const southEast = offsetCoordinate(plan.center.latitude, plan.center.longitude, cell.south_m, cell.east_m);
      const southWest = offsetCoordinate(plan.center.latitude, plan.center.longitude, cell.south_m, cell.west_m);
      areas.push({
        type: "Feature",
        properties: { cell_id: cell.cell_id },
        geometry: { type: "Polygon", coordinates: [[northWest, northEast, southEast, southWest, northWest]] }
      });
    }
    for (const line of plan.survey_lines) {
      lines.push({
        type: "Feature",
        properties: { cell_id: line.cell_id },
        geometry: {
          type: "LineString",
          coordinates: [
            [line.start_longitude, line.start_latitude],
            [line.end_longitude, line.end_latitude]
          ]
        }
      });
    }
    points.push({
      type: "Feature",
      properties: { kind: "home" },
      geometry: { type: "Point", coordinates: [plan.home.longitude, plan.home.latitude] }
    });
  }
  if (selected) {
    points.push({
      type: "Feature",
      properties: { kind: "center" },
      geometry: { type: "Point", coordinates: [selected.longitude, selected.latitude] }
    });
  }
  return {
    areas: { type: "FeatureCollection", features: areas } as GeoJSON.FeatureCollection,
    lines: { type: "FeatureCollection", features: lines } as GeoJSON.FeatureCollection,
    points: { type: "FeatureCollection", features: points } as GeoJSON.FeatureCollection
  };
}

function addMissionLayers(map: Map) {
  if (map.getSource("mission_cells")) return;
  map.addSource("mission_cells", { type: "geojson", data: emptyCollection });
  map.addSource("mission_lines", { type: "geojson", data: emptyCollection });
  map.addSource("mission_points", { type: "geojson", data: emptyCollection });
  map.addLayer({
    id: "mission_cell_fill",
    type: "fill",
    source: "mission_cells",
    paint: { "fill-color": "#2dd4bf", "fill-opacity": 0.10 }
  });
  map.addLayer({
    id: "mission_cell_outline",
    type: "line",
    source: "mission_cells",
    paint: { "line-color": "#5eead4", "line-width": 2 }
  });
  map.addLayer({
    id: "mission_survey_lines",
    type: "line",
    source: "mission_lines",
    paint: { "line-color": "#fbbf24", "line-width": 2, "line-opacity": 0.9 }
  });
  map.addLayer({
    id: "mission_home",
    type: "circle",
    source: "mission_points",
    filter: ["==", ["get", "kind"], "home"],
    paint: { "circle-radius": 7, "circle-color": "#60a5fa", "circle-stroke-color": "#071014", "circle-stroke-width": 2 }
  });
  map.addLayer({
    id: "mission_center",
    type: "circle",
    source: "mission_points",
    filter: ["==", ["get", "kind"], "center"],
    paint: { "circle-radius": 7, "circle-color": "#5eead4", "circle-stroke-color": "#071014", "circle-stroke-width": 2 }
  });
}

export function TerrainMissionMap({
  selected,
  plan,
  onSelect
}: {
  selected: { latitude: number; longitude: number } | null;
  plan: MapPlan | null;
  onSelect: (latitude: number, longitude: number) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<Map | null>(null);
  const onSelectRef = useRef(onSelect);
  const initialSelectedRef = useRef(selected);
  onSelectRef.current = onSelect;

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const initialSelected = initialSelectedRef.current;
    const initialCenter: [number, number] = initialSelected
      ? [initialSelected.longitude, initialSelected.latitude]
      : [0, 20];
    const map = new Map({
      container: containerRef.current,
      style: mapStyle(),
      center: initialCenter,
      zoom: initialSelected ? 13 : 1.5,
      attributionControl: {},
      maxPitch: 70
    });
    map.addControl(new NavigationControl({ visualizePitch: true }), "top-right");
    map.on("click", (event: MapMouseEvent) => onSelectRef.current(event.lngLat.lat, event.lngLat.lng));
    map.on("style.load", () => {
      addMissionLayers(map);
      if (map.getSource("terrain_dem")) map.setTerrain({ source: "terrain_dem", exaggeration: 1.25 });
    });
    mapRef.current = map;
    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const geometry = planGeometry(plan, selected);
    const update = () => {
      addMissionLayers(map);
      (map.getSource("mission_cells") as GeoJSONSource | undefined)?.setData(geometry.areas);
      (map.getSource("mission_lines") as GeoJSONSource | undefined)?.setData(geometry.lines);
      (map.getSource("mission_points") as GeoJSONSource | undefined)?.setData(geometry.points);
      if (plan) {
        const bounds = new LngLatBounds();
        for (const cell of plan.cells) {
          bounds.extend(offsetCoordinate(plan.center.latitude, plan.center.longitude, cell.north_m, cell.west_m));
          bounds.extend(offsetCoordinate(plan.center.latitude, plan.center.longitude, cell.south_m, cell.east_m));
        }
        map.fitBounds(bounds, { padding: 48, duration: 500, maxZoom: 19 });
      } else if (selected) {
        map.easeTo({ center: [selected.longitude, selected.latitude], zoom: Math.max(map.getZoom(), 13), duration: 350 });
      }
    };
    if (map.isStyleLoaded()) update();
    else map.once("style.load", update);
  }, [plan, selected]);

  return <div ref={containerRef} className="min-h-[420px] flex-1" aria-label="Interactive terrain mission map" />;
}
