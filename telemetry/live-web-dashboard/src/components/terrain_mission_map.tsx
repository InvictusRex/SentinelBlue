import { useEffect, useRef, useState } from "react";
import { Map, Marker, LngLatBounds, NavigationControl, setWorkerUrl, type GeoJSONSource, type StyleSpecification } from "maplibre-gl";
import mapLibreWorkerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?url";
import type { Position, SurveyPlan } from "../lib/survey_plan";

setWorkerUrl(mapLibreWorkerUrl);
export type DrawMode = "navigate" | "rectangle" | "polygon" | "home" | "center";
interface Props {
  boundary: Position[]; home: Position | null; plan: SurveyPlan | null; sortie: number;
  center?: Position | null;
  mode: DrawMode; focus: Position; fitVersion: number;
  onClick: (point: Position) => void; onMoveVertex: (index: number, point: Position) => void;
}
const coordinates = (point: Position): [number, number] => [point.longitude, point.latitude];
const collection = (features: GeoJSON.Feature[]): GeoJSON.FeatureCollection => ({ type: "FeatureCollection", features });
const line = (points: Position[], properties = {}): GeoJSON.Feature => ({ type: "Feature", properties, geometry: { type: "LineString", coordinates: points.map(coordinates) } });

function mapStyle(): StyleSpecification | string {
  const key = (import.meta as ImportMeta & { env?: Record<string, string> }).env?.VITE_MAPTILER_KEY;
  if (key) return `https://api.maptiler.com/maps/hybrid/style.json?key=${encodeURIComponent(key)}`;
  return { version: 8, sources: {
    tiles: { type: "raster", tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"], tileSize: 256, maxzoom: 19, attribution: "© OpenStreetMap contributors" },
    dem: { type: "raster-dem", url: "https://demotiles.maplibre.org/terrain-tiles/tiles.json", tileSize: 256 }
  }, layers: [{ id: "tiles", type: "raster", source: "tiles" }, { id: "terrain", type: "hillshade", source: "dem", paint: { "hillshade-exaggeration": 0.25 } }] };
}

export function TerrainMissionMap(props: Props) {
  const { boundary, home, plan, sortie, mode, center } = props;
  const container = useRef<HTMLDivElement>(null);
  const mapRef = useRef<Map | null>(null);
  const current = useRef(props); current.current = props;
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState(false);
  useEffect(() => {
    if (!container.current) return;
    setLoaded(false);
    setError(false);
    const map = new Map({ container: container.current, style: mapStyle(), center: coordinates(current.current.focus), zoom: 17, maxPitch: 0 });
    mapRef.current = map;
    map.addControl(new NavigationControl({ showCompass: false }), "top-right");
    map.doubleClickZoom.disable();
    map.on("click", event => current.current.onClick({ latitude: event.lngLat.lat, longitude: event.lngLat.lng }));
    map.on("error", () => setError(true));
    map.on("style.load", () => {
      for (const source of ["area", "edge", "grid", "transit"]) map.addSource(source, { type: "geojson", data: collection([]) });
      map.addLayer({ id: "area", type: "fill", source: "area", paint: { "fill-color": "#2dd4bf", "fill-opacity": 0.14 } });
      map.addLayer({ id: "edge", type: "line", source: "edge", paint: { "line-color": "#0d9488", "line-width": 3 } });
      map.addLayer({ id: "transit", type: "line", source: "transit", paint: { "line-color": "#2563eb", "line-width": 2, "line-dasharray": [3, 3] } });
      map.addLayer({ id: "grid", type: "line", source: "grid", paint: { "line-color": ["case", ["get", "active"], "#b45309", "#64748b"], "line-width": ["case", ["get", "active"], 3, 1], "line-opacity": ["case", ["get", "active"], 1, 0.4] } });
      setLoaded(true);
    });
    const observer = new ResizeObserver(() => map.resize()); observer.observe(container.current);
    return () => { observer.disconnect(); map.remove(); mapRef.current = null; };
  }, []);
  useEffect(() => {
    const map = mapRef.current;
    if (!loaded || !map || !map.getSource("area")) return;
    const closed = boundary.length >= 3 ? [...boundary, boundary[0]] : boundary;
    (map.getSource("area") as GeoJSONSource).setData(collection(boundary.length >= 3 ? [{ type: "Feature", properties: {}, geometry: { type: "Polygon", coordinates: [closed.map(coordinates)] } }] : []));
    (map.getSource("edge") as GeoJSONSource).setData(collection(boundary.length >= 2 ? [line(closed)] : []));
    (map.getSource("grid") as GeoJSONSource).setData(collection((plan?.survey_lines ?? []).map(track => line([
      { latitude: track.start_latitude, longitude: track.start_longitude }, { latitude: track.end_latitude, longitude: track.end_longitude }
    ], { active: track.sortie_id === sortie }))));
    const selected = plan?.sorties.find(item => item.sortie_id === sortie);
    const transits = selected ? selected.route.flatMap((point, index, route) => index % 2 === 0 && index + 1 < route.length ? [line([point, route[index + 1]])] : []) : [];
    (map.getSource("transit") as GeoJSONSource).setData(collection(transits));
    const markers = (mode === "center" ? [] : boundary).map((point, index) => {
      const element = document.createElement("button"); element.type = "button"; element.textContent = String(index + 1);
      element.className = "survey-vertex"; element.setAttribute("aria-label", `Boundary vertex ${index + 1}; drag to move. Coordinates can also be edited in Area.`);
      element.addEventListener("click", event => event.stopPropagation());
      const marker = new Marker({ element, draggable: mode === "navigate" }).setLngLat(coordinates(point)).addTo(map);
      marker.on("dragend", () => { const p = marker.getLngLat(); current.current.onMoveVertex(index, { latitude: p.lat, longitude: p.lng }); });
      return marker;
    });
    for (const [point, label, color] of [[center, "C", "#0f766e"], [home ?? plan?.home, "H", "#2563eb"], [selected?.route[1], "S", "#15803d"], [selected ? selected.route[selected.route.length - 2] : undefined, "E", "#b45309"]] as const) {
      if (!point) continue;
      const element = document.createElement("span"); element.className = "survey-vertex";
      element.style.background = color; element.textContent = label; element.title = label === "C" ? "Search centre" : label === "H" ? "Launch and landing" : label === "S" ? "First survey line" : "Last survey line";
      markers.push(new Marker({ element }).setLngLat(coordinates(point)).addTo(map));
    }
    return () => markers.forEach(marker => marker.remove());
  }, [boundary, home, plan, sortie, mode, center, loaded]);
  useEffect(() => {
    const map = mapRef.current;
    if (map) map.getCanvas().style.cursor = props.mode === "navigate" ? "" : "crosshair";
  }, [props.mode]);
  useEffect(() => {
    mapRef.current?.easeTo({ center: coordinates(props.focus), zoom: 17, duration: 300 });
  }, [props.focus]);
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const { boundary, home, focus } = current.current;
    if (boundary.length > 1) {
      const bounds = new LngLatBounds();
      [...boundary, ...(home ? [home] : [])].forEach(point => bounds.extend(coordinates(point)));
      map.fitBounds(bounds, { padding: 60, maxZoom: 20, duration: 300 });
    } else map.easeTo({ center: coordinates(focus), zoom: 17, duration: 300 });
  }, [props.fitVersion]);
  return <div className="relative min-h-[420px] flex-1"><div ref={container} style={{ position: "absolute", inset: 0 }} aria-label="Survey area map" />
    {error && <div role="status" className="absolute bottom-20 left-2 right-2 rounded bg-hud-bg/95 px-3 py-2 text-xs text-hud-muted sm:bottom-8">Some map tiles could not load. You can still enter boundary coordinates in Area.</div>}
  </div>;
}
