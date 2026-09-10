/* ======================================================================================
   Drone Mission Control & Autonomous Mission Planner App (app.js)
   ====================================================================================== */

document.addEventListener('DOMContentLoaded', () => {
  // 1. DOM ELEMENTS & STATE
  const el = {
    batV: document.getElementById('val-bat-v'),
    batA: document.getElementById('val-bat-a'),
    batW: document.getElementById('val-bat-w'),
    batPct: document.getElementById('val-bat-pct'),
    batBar: document.getElementById('bat-remaining-bar'),
    altRel: document.getElementById('val-alt-rel'),
    altMsl: document.getElementById('val-alt-msl'),
    climb: document.getElementById('val-climb'),
    heading: document.getElementById('val-heading'),
    headingCard: document.getElementById('val-heading-cardinal'),
    roll: document.getElementById('val-roll'),
    pitch: document.getElementById('val-pitch'),
    rssi: document.getElementById('val-rssi'),
    rssiBar: document.getElementById('rssi-bar'),
    sbcTemp: document.getElementById('val-sbc-temp'),
    sbcLoad: document.getElementById('val-sbc-load'),
    badgeArm: document.getElementById('badge-arm'),
    badgeMode: document.getElementById('badge-mode'),
    badgeMission: document.getElementById('badge-mission'),
    connIndicator: document.getElementById('conn-indicator'),
    connText: document.getElementById('conn-text'),
    gpsFix: document.getElementById('badge-gps-fix'),
    gpsSats: document.getElementById('badge-gps-sats'),
    gpsHdop: document.getElementById('badge-gps-hdop'),
    lat: document.getElementById('val-lat'),
    lon: document.getElementById('val-lon'),
    accX: document.getElementById('val-acc-x'),
    accY: document.getElementById('val-acc-y'),
    accZ: document.getElementById('val-acc-z'),
    cellDelta: document.getElementById('val-cell-delta'),
    errRoll: document.getElementById('val-err-roll'),
    errPitch: document.getElementById('val-err-pitch'),
    sbcTempTxt: document.getElementById('sbc-temp-txt'),
    sbcCpuTxt: document.getElementById('sbc-cpu-txt'),
    sbcRamTxt: document.getElementById('sbc-ram-txt'),
    sbcDiskTxt: document.getElementById('sbc-disk-txt'),
    sbcUptimeTxt: document.getElementById('sbc-uptime-txt'),

    // Map HUD elements
    hudDot: document.getElementById('hud-dot'),
    hudMissionState: document.getElementById('hud-mission-state'),
    hudWpTracker: document.getElementById('hud-wp-tracker'),
    hudWpDist: document.getElementById('hud-wp-dist'),
    hudProgressFill: document.getElementById('hud-progress-fill'),
    btnHudStart: document.getElementById('btn-hud-start'),
    btnHudPause: document.getElementById('btn-hud-pause'),
    btnHudRtl: document.getElementById('btn-hud-rtl'),

    // Modal & Planner elements
    gridModal: document.getElementById('grid-modal'),
    btnOpenModal: document.getElementById('btn-open-grid-modal'),
    btnCloseModal: document.getElementById('btn-close-modal'),
    btnCancelModal: document.getElementById('btn-cancel-modal'),
    btnPickCenter: document.getElementById('btn-pick-center'),
    radiusSlider: document.getElementById('input-radius-slider'),
    radiusLabel: document.getElementById('val-radius-label'),
    inputSpacing: document.getElementById('input-spacing'),
    inputAngle: document.getElementById('input-angle'),
    inputAltitude: document.getElementById('input-altitude'),
    inputSpeed: document.getElementById('input-speed'),
    inputEndAction: document.getElementById('input-end-action'),
    inputCustomLat: document.getElementById('input-custom-lat'),
    inputCustomLon: document.getElementById('input-custom-lon'),
    statWpCount: document.getElementById('stat-wp-count'),
    statTotalDist: document.getElementById('stat-total-dist'),
    statEstTime: document.getElementById('stat-est-time'),
    btnPreviewMission: document.getElementById('btn-preview-mission'),
    btnSubmitGrid: document.getElementById('btn-submit-grid'),
    btnModalStart: document.getElementById('btn-modal-start'),
    btnModalPause: document.getElementById('btn-modal-pause'),
    btnModalAbort: document.getElementById('btn-modal-abort'),
    btnModalClear: document.getElementById('btn-modal-clear'),
    alertBox: document.getElementById('mission-alert-box')
  };

  const motors = [1, 2, 3, 4].map(i => ({
    pct: document.getElementById(`val-m${i}-pct`),
    bar: document.getElementById(`bar-m${i}`),
    pwm: document.getElementById(`val-m${i}-pwm`)
  }));

  const cells = [1, 2, 3, 4, 5, 6].map(i => ({
    val: document.getElementById(`val-cell-${i}`),
    bar: document.getElementById(`cell-bar-${i}`)
  }));

  let currentTelemetry = {};
  let isPickingPointOnMap = false;
  let selectedCenter = [12.971598, 77.594562];
  let activeWpMarkers = [];

  // 2. LEAFLET MAP & MISSION LAYERS
  let map, droneMarker, flightPathPolyline;
  let searchCenterMarker, searchRadiusCircle, missionRoutePolyline, missionLayerGroup;
  const flightHistory = [];
  const initialCoords = [12.971598, 77.594562];

  function initMap() {
    map = L.map('leaflet-map', {
      center: initialCoords,
      zoom: 18,
      zoomControl: true,
      attributionControl: false
    });

    // Google Maps Tile Layers (Hybrid, Terrain, Roadmap, Satellite)
    const googleHybrid = L.tileLayer('https://{s}.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', {
      maxZoom: 21,
      subdomains: ['mt0', 'mt1', 'mt2', 'mt3']
    });

    const googleTerrain = L.tileLayer('https://{s}.google.com/vt/lyrs=p&x={x}&y={y}&z={z}', {
      maxZoom: 21,
      subdomains: ['mt0', 'mt1', 'mt2', 'mt3']
    });

    const googleRoadmap = L.tileLayer('https://{s}.google.com/vt/lyrs=m&x={x}&y={y}&z={z}', {
      maxZoom: 21,
      subdomains: ['mt0', 'mt1', 'mt2', 'mt3']
    });

    const googleSatellite = L.tileLayer('https://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', {
      maxZoom: 21,
      subdomains: ['mt0', 'mt1', 'mt2', 'mt3']
    });

    const osm = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19
    });

    // Default layer: Google Hybrid (Satellite + Buildings + Streets)
    googleHybrid.addTo(map);

    // Layer Switcher Control
    const baseMaps = {
      "🛰️ Google Hybrid (Satellite + Buildings & Roads)": googleHybrid,
      "⛰️ Google Terrain (Elevation & Relief)": googleTerrain,
      "🏙️ Google Roadmap (Streets & Buildings)": googleRoadmap,
      "🌍 Google Satellite (Pure Imagery)": googleSatellite,
      "🗺️ OpenStreetMap": osm
    };

    L.control.layers(baseMaps, null, { position: 'topleft', collapsed: true }).addTo(map);

    const droneIcon = L.divIcon({
      className: 'drone-leaflet-icon',
      html: `<div id="drone-map-svg" style="transform: rotate(0deg); transition: transform 0.1s linear;">
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#00f0ff" stroke-width="2">
                <polygon points="12 2 19 21 12 17 5 21 12 2" fill="rgba(0, 240, 255, 0.4)"/>
              </svg>
             </div>`,
      iconSize: [32, 32],
      iconAnchor: [16, 16]
    });

    droneMarker = L.marker(initialCoords, { icon: droneIcon }).addTo(map);

    flightPathPolyline = L.polyline([], {
      color: '#00f0ff',
      weight: 3,
      opacity: 0.8,
      dashArray: '4, 6'
    }).addTo(map);

    // Mission Preview & Autonomy Layers
    missionLayerGroup = L.layerGroup().addTo(map);

    searchRadiusCircle = L.circle(initialCoords, {
      radius: 50,
      color: '#ffd000',
      weight: 2,
      dashArray: '6, 6',
      fillColor: '#ffd000',
      fillOpacity: 0.08
    }).addTo(map);

    // Map Click Listener for Target Point Picking
    map.on('click', (e) => {
      if (isPickingPointOnMap) {
        setSearchCenter(e.latlng.lat, e.latlng.lng);
        togglePickMode(false);
        el.gridModal.classList.remove('hidden');
        previewMissionRoute();
      }
    });
  }

  function setSearchCenter(lat, lon) {
    selectedCenter = [lat, lon];
    el.inputCustomLat.value = lat.toFixed(6);
    el.inputCustomLon.value = lon.toFixed(6);

    const r = parseFloat(el.radiusSlider.value) || 50;
    searchRadiusCircle.setLatLng(selectedCenter);
    searchRadiusCircle.setRadius(r);

    if (searchCenterMarker) {
      searchCenterMarker.setLatLng(selectedCenter);
    } else {
      const centerIcon = L.divIcon({
        className: 'center-marker-icon',
        html: '🎯',
        iconSize: [24, 24],
        iconAnchor: [12, 12]
      });
      searchCenterMarker = L.marker(selectedCenter, { icon: centerIcon }).addTo(map);
    }
  }

  function togglePickMode(enable) {
    isPickingPointOnMap = (enable !== undefined) ? enable : !isPickingPointOnMap;
    if (isPickingPointOnMap) {
      el.btnPickCenter.textContent = '❌ Cancel Picking (Click Map)';
      el.btnPickCenter.style.borderColor = 'var(--accent-red)';
      el.btnPickCenter.style.color = 'var(--accent-red)';
      map.getContainer().style.cursor = 'crosshair';
      el.gridModal.classList.add('hidden');
    } else {
      el.btnPickCenter.textContent = '📍 Pick Center on Map';
      el.btnPickCenter.style.borderColor = '';
      el.btnPickCenter.style.color = '';
      map.getContainer().style.cursor = '';
      isPickingPointOnMap = false;
    }
  }

  initMap();

  document.getElementById('btn-recenter').addEventListener('click', () => {
    if (droneMarker) {
      map.panTo(droneMarker.getLatLng(), { animate: true });
    }
  });

  el.btnPickCenter.addEventListener('click', () => togglePickMode());

  // 3. CHART.JS REAL-TIME CHARTS
  const maxDataPoints = 40;
  const timeLabels = Array(maxDataPoints).fill('');

  const gyroCtx = document.getElementById('gyroChart').getContext('2d');
  const gyroChart = new Chart(gyroCtx, {
    type: 'line',
    data: {
      labels: [...timeLabels],
      datasets: [
        { label: 'Roll Rate (ωx)', data: Array(maxDataPoints).fill(0), borderColor: '#00f0ff', backgroundColor: 'transparent', borderWidth: 2, pointRadius: 0, tension: 0.2 },
        { label: 'Pitch Rate (ωy)', data: Array(maxDataPoints).fill(0), borderColor: '#ffd000', backgroundColor: 'transparent', borderWidth: 2, pointRadius: 0, tension: 0.2 },
        { label: 'Yaw Rate (ωz)', data: Array(maxDataPoints).fill(0), borderColor: '#ff0077', backgroundColor: 'transparent', borderWidth: 2, pointRadius: 0, tension: 0.2 }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { display: false },
        y: { grid: { color: '#1e293f' }, ticks: { color: '#8b9bb4', font: { size: 10 } } }
      }
    }
  });

  const pidCtx = document.getElementById('pidChart').getContext('2d');
  const pidChart = new Chart(pidCtx, {
    type: 'line',
    data: {
      labels: [...timeLabels],
      datasets: [
        { label: 'Target Pitch', data: Array(maxDataPoints).fill(0), borderColor: '#ffd000', borderWidth: 1.5, borderDash: [4, 4], pointRadius: 0, tension: 0.2 },
        { label: 'Measured Pitch', data: Array(maxDataPoints).fill(0), borderColor: '#00ff88', borderWidth: 2, pointRadius: 0, tension: 0.2 },
        { label: 'Target Roll', data: Array(maxDataPoints).fill(0), borderColor: '#00f0ff', borderWidth: 1.5, borderDash: [4, 4], pointRadius: 0, tension: 0.2 },
        { label: 'Measured Roll', data: Array(maxDataPoints).fill(0), borderColor: '#ff0077', borderWidth: 2, pointRadius: 0, tension: 0.2 }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { display: false },
        y: { grid: { color: '#1e293f' }, ticks: { color: '#8b9bb4', font: { size: 10 } } }
      }
    }
  });

  // 4. TELEMETRY DISPATCHER & LIVE MISSION TRACKING
  function updateTelemetry(data) {
    currentTelemetry = data;

    const rawV = data.battery_voltage || 0;
    const isUsb = rawV < 0.5;
    const v = isUsb ? '5.00 (USB)' : rawV.toFixed(2);
    const a = (data.battery_current || 0).toFixed(1);
    const w = isUsb ? '0' : (data.battery_voltage * data.battery_current).toFixed(0);
    const pct = isUsb ? 100 : (data.battery_remaining || 0);

    el.batV.textContent = v;
    el.batA.textContent = a;
    el.batW.textContent = w;
    el.batPct.textContent = isUsb ? 'USB' : pct;
    el.batBar.style.width = `${pct}%`;
    if (isUsb) el.batBar.style.background = 'var(--accent-blue)';

    const displayedAlt = (data.altitude_relative !== 0 ? data.altitude_relative : (data.altitude_msl || 0));
    el.altRel.textContent = (displayedAlt || 0).toFixed(1);
    el.altMsl.textContent = (data.altitude_msl || displayedAlt || 0).toFixed(1);
    el.climb.textContent = (data.climb_rate >= 0 ? '+' : '') + (data.climb_rate || 0).toFixed(1);

    const hdg = Math.round(data.heading || 0);
    el.heading.textContent = hdg.toString().padStart(3, '0');
    el.headingCard.textContent = getCardinal(hdg);
    el.roll.textContent = (data.attitude_roll || 0).toFixed(1);
    el.pitch.textContent = (data.attitude_pitch || 0).toFixed(1);

    const rssi = data.rc_rssi || 0;
    el.rssi.textContent = rssi;
    el.rssiBar.style.width = `${rssi}%`;
    el.sbcTemp.textContent = (data.sbc?.cpu_temp_c || 48.0).toFixed(1);
    el.sbcLoad.textContent = Math.round(data.sbc?.cpu_load_percent || 0);

    const nowSec = Date.now() / 1000;
    const isHeartbeatLost = !data.connected || (data.last_packet_timestamp > 0 && (nowSec - data.last_packet_timestamp > 10.0));

    if (data.armed && !isHeartbeatLost) {
      el.badgeArm.textContent = 'ARMED';
      el.badgeArm.className = 'status-badge armed';
    } else {
      el.badgeArm.textContent = 'DISARMED';
      el.badgeArm.className = 'status-badge disarmed';
    }

    if (isHeartbeatLost) {
      el.badgeMode.textContent = 'NO HEARTBEAT';
      el.badgeMode.className = 'status-badge lost';
      el.connText.textContent = 'NO HEARTBEAT (>10s)';
      el.connIndicator.className = 'conn-indicator lost';
    } else {
      el.badgeMode.textContent = data.flight_mode || 'STABILIZE';
      el.badgeMode.className = 'status-badge mode-badge';
      el.connText.textContent = 'PIXHAWK LIVE';
      el.connIndicator.className = 'conn-indicator online';
    }
    el.badgeMission.textContent = isHeartbeatLost ? 'LINK LOST' : (data.mission_state || 'STANDBY');

    // Live Map HUD Update
    if (el.hudMissionState) {
      el.hudMissionState.textContent = `MISSION: ${data.mission_state || 'STANDBY'}`;
      const isAuto = (data.flight_mode === 'AUTO');
      const isLoiter = (data.flight_mode === 'LOITER');
      el.hudDot.className = 'hud-dot' + (isAuto ? ' active' : (isLoiter ? ' paused' : ''));
      const currSeq = data.mission_current_seq || 0;
      const totItems = data.mission_total_items || 0;
      el.hudWpTracker.textContent = totItems > 0 ? `WP ${currSeq} / ${totItems}` : '-- / --';
      el.hudProgressFill.style.width = `${data.mission_progress_percent || 0}%`;

      // Update Waypoint Marker active/completed classes
      activeWpMarkers.forEach((m, idx) => {
        const markerDom = m.getElement();
        if (markerDom) {
          const wpIdx = idx + 1;
          if (wpIdx < currSeq) {
            markerDom.className = 'leaflet-marker-icon wp-marker-icon completed';
          } else if (wpIdx === currSeq) {
            markerDom.className = 'leaflet-marker-icon wp-marker-icon active';
          } else {
            markerDom.className = 'leaflet-marker-icon wp-marker-icon';
          }
        }
      });
    }

    el.gpsFix.textContent = data.gps_fix_type || '3D FIX';
    el.gpsSats.textContent = `${data.satellites || 0} SATS`;
    el.gpsHdop.textContent = `HDOP: ${(data.hdop || 1.0).toFixed(1)}`;

    if (data.latitude && data.longitude && data.latitude !== 0) {
      el.lat.textContent = data.latitude.toFixed(6);
      el.lon.textContent = data.longitude.toFixed(6);

      const newLatLng = [data.latitude, data.longitude];
      droneMarker.setLatLng(newLatLng);

      const svgEl = document.getElementById('drone-map-svg');
      if (svgEl) svgEl.style.transform = `rotate(${hdg}deg)`;

      flightHistory.push(newLatLng);
      if (flightHistory.length > 300) flightHistory.shift();
      flightPathPolyline.setLatLngs(flightHistory);
    }

    el.accX.textContent = (data.accel_x || 0).toFixed(2);
    el.accY.textContent = (data.accel_y || 0).toFixed(2);
    el.accZ.textContent = (data.accel_z || 1.0).toFixed(2);

    gyroChart.data.datasets[0].data.shift();
    gyroChart.data.datasets[0].data.push(data.gyro_x || 0);
    gyroChart.data.datasets[1].data.shift();
    gyroChart.data.datasets[1].data.push(data.gyro_y || 0);
    gyroChart.data.datasets[2].data.shift();
    gyroChart.data.datasets[2].data.push(data.gyro_z || 0);
    gyroChart.update();

    pidChart.data.datasets[0].data.shift();
    pidChart.data.datasets[0].data.push(data.target_pitch || 0);
    pidChart.data.datasets[1].data.shift();
    pidChart.data.datasets[1].data.push(data.attitude_pitch || 0);
    pidChart.data.datasets[2].data.shift();
    pidChart.data.datasets[2].data.push(data.target_roll || 0);
    pidChart.data.datasets[3].data.shift();
    pidChart.data.datasets[3].data.push(data.attitude_roll || 0);
    pidChart.update();

    el.errRoll.textContent = `${(data.error_roll || 0).toFixed(1)}°`;
    el.errPitch.textContent = `${(data.error_pitch || 0).toFixed(1)}°`;

    const cellVals = data.cell_voltages || [];
    cells.forEach((c, idx) => {
      const cv = cellVals[idx] || 0.0;
      if (cv > 1.0) {
        c.val.textContent = `${cv.toFixed(2)}V`;
        const p = Math.max(0, Math.min(100, ((cv - 3.2) / 1.0) * 100));
        c.bar.style.width = `${p}%`;
        c.bar.style.background = cv >= 3.7 ? 'var(--accent-green)' : (cv >= 3.5 ? 'var(--accent-yellow)' : 'var(--accent-red)');
      } else {
        c.val.textContent = '--.--';
        c.bar.style.width = '0%';
      }
    });

    const delta = data.cell_delta_mv || 0;
    el.cellDelta.textContent = `ΔV: ${delta} mV (${delta <= 30 ? 'BALANCED ✓' : 'UNBALANCED !'})`;
    el.cellDelta.style.color = delta <= 30 ? 'var(--accent-green)' : 'var(--accent-red)';

    const pwms = data.motor_pwm || [1000, 1000, 1000, 1000];
    const pcts = data.motor_percent || [0, 0, 0, 0];
    motors.forEach((m, idx) => {
      const p = pcts[idx] || 0;
      m.pct.textContent = `${p}%`;
      m.bar.style.height = `${p}%`;
      m.pwm.textContent = `${pwms[idx] || 1000}µs`;
    });

    if (data.sbc) {
      el.sbcTempTxt.innerHTML = `${data.sbc.cpu_temp_c.toFixed(1)} °C <span class="${data.sbc.cpu_temp_c > 75 ? 'badge-warn' : 'badge-ok'}">${data.sbc.cpu_temp_c > 75 ? 'HIGH' : 'NORMAL'}</span>`;
      el.sbcCpuTxt.textContent = `${data.sbc.cpu_load_percent.toFixed(1)} %`;
      el.sbcRamTxt.textContent = `${data.sbc.ram_used_mb} MB / ${data.sbc.ram_total_mb} MB (${data.sbc.ram_percent.toFixed(1)}%)`;
      el.sbcDiskTxt.textContent = `${data.sbc.disk_free_gb} GB (${data.sbc.disk_percent.toFixed(0)}% used)`;
      const hrs = Math.floor(data.sbc.uptime_seconds / 3600);
      const mins = Math.floor((data.sbc.uptime_seconds % 3600) / 60);
      el.sbcUptimeTxt.textContent = `${hrs}h ${mins}m`;
    }
  }

  function getCardinal(deg) {
    const directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
    return directions[Math.round(deg / 45) % 8];
  }

  // 5. AUTONOMOUS MISSION PLANNER & MODAL LOGIC
  el.btnOpenModal.addEventListener('click', () => {
    el.gridModal.classList.remove('hidden');
    el.alertBox.classList.add('hidden');
    previewMissionRoute();
  });

  const closeModal = () => {
    el.gridModal.classList.add('hidden');
    togglePickMode(false);
  };
  el.btnCloseModal.addEventListener('click', closeModal);
  el.btnCancelModal.addEventListener('click', closeModal);

  // Radius slider live update
  el.radiusSlider.addEventListener('input', () => {
    const r = parseFloat(el.radiusSlider.value) || 50;
    el.radiusLabel.textContent = r;
    searchRadiusCircle.setRadius(r);
  });
  el.radiusSlider.addEventListener('change', () => previewMissionRoute());

  // Input changes trigger route preview
  [el.inputSpacing, el.inputAngle, el.inputAltitude, el.inputSpeed, el.inputEndAction].forEach(elem => {
    elem.addEventListener('change', () => previewMissionRoute());
  });

  // Pattern radio selection
  document.querySelectorAll('input[name="mission-pattern"]').forEach(radio => {
    radio.addEventListener('change', () => previewMissionRoute());
  });

  // Center mode radio selection
  document.querySelectorAll('input[name="center-mode"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
      const mode = e.target.value;
      if (mode === 'current') {
        const droneLat = currentTelemetry.latitude || 12.971598;
        const droneLon = currentTelemetry.longitude || 77.594562;
        setSearchCenter(droneLat, droneLon);
        previewMissionRoute();
      } else if (mode === 'map') {
        togglePickMode(true);
      }
    });
  });

  function getSelectedMissionParams() {
    const pattern = document.querySelector('input[name="mission-pattern"]:checked')?.value || 'grid';
    const radius = parseFloat(el.radiusSlider.value) || 50;
    const spacing = parseFloat(el.inputSpacing.value) || 10;
    const angle = parseFloat(el.inputAngle.value) || 0;
    const altitude = parseFloat(el.inputAltitude.value) || 15;
    const speed = parseFloat(el.inputSpeed.value) || 5;
    const end_action = el.inputEndAction.value || 'RTL';

    let lat = selectedCenter[0];
    let lon = selectedCenter[1];

    const centerMode = document.querySelector('input[name="center-mode"]:checked')?.value || 'current';
    if (centerMode === 'custom') {
      lat = parseFloat(el.inputCustomLat.value) || lat;
      lon = parseFloat(el.inputCustomLon.value) || lon;
    }

    return { lat, lon, radius, spacing, angle, altitude, speed, pattern, end_action };
  }

  // 6. ROUTE PREVIEW ON MAP
  async function previewMissionRoute() {
    const params = getSelectedMissionParams();
    try {
      const res = await fetch('/api/mission/preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
      });
      const data = await res.json();
      if (data && data.waypoints) {
        renderMissionOnMap(data);
        el.statWpCount.textContent = `${data.waypoint_count} WPs`;
        el.statTotalDist.textContent = `${data.total_distance_m} m`;
        const mins = Math.floor(data.estimated_duration_s / 60);
        const secs = Math.floor(data.estimated_duration_s % 60);
        el.statEstTime.textContent = `${mins}m ${secs.toString().padStart(2, '0')}s`;
      }
    } catch (e) {
      console.error('Preview failed:', e);
    }
  }

  function renderMissionOnMap(plan) {
    missionLayerGroup.clearLayers();
    activeWpMarkers = [];

    const waypoints = plan.waypoints || [];
    if (waypoints.length === 0) return;

    // Draw waypoints & markers
    const latLngs = waypoints.map(wp => [wp[0], wp[1]]);

    // Path Polyline
    missionRoutePolyline = L.polyline(latLngs, {
      color: '#ffd000',
      weight: 2.5,
      dashArray: '6, 6',
      opacity: 0.9
    }).addTo(missionLayerGroup);

    waypoints.forEach((wp, idx) => {
      const wpIcon = L.divIcon({
        className: 'wp-marker-icon',
        html: `<span>${idx + 1}</span>`,
        iconSize: [22, 22],
        iconAnchor: [11, 11]
      });
      const marker = L.marker([wp[0], wp[1]], { icon: wpIcon })
        .bindPopup(`<b>Waypoint ${idx + 1}</b><br>Alt: ${wp[2]}m AGL<br>Lat: ${wp[0].toFixed(6)}<br>Lon: ${wp[1].toFixed(6)}`)
        .addTo(missionLayerGroup);
      activeWpMarkers.push(marker);
    });

    // Takeoff Dynamic Point Marker
    const takeoffIcon = L.divIcon({
      className: 'takeoff-marker-icon',
      html: '<span>🛫</span>',
      iconSize: [26, 26],
      iconAnchor: [13, 13]
    });
    L.marker([plan.center[0], plan.center[1]], { icon: takeoffIcon })
      .bindPopup(`<b>Vertical Takeoff Target Datum</b><br>Ascends to ${plan.altitude_m}m AGL before cruising to WP 1`)
      .addTo(missionLayerGroup);
  }

  el.btnPreviewMission.addEventListener('click', previewMissionRoute);

  // 7. UPLOAD & MISSION CONTROL ACTIONS
  el.btnSubmitGrid.addEventListener('click', async () => {
    const params = getSelectedMissionParams();
    el.alertBox.className = 'alert-box';
    el.alertBox.textContent = 'Uploading mission to Pixhawk via MAVLink...';
    el.alertBox.classList.remove('hidden');

    try {
      const res = await fetch('/api/mission/upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
      });
      const resData = await res.json();
      if (res.ok && resData.status === 'success') {
        el.alertBox.className = 'alert-box success';
        el.alertBox.textContent = `✓ Mission Uploaded! Generated ${resData.plan?.waypoint_count || '--'} waypoints with dynamic vertical takeoff. Ready to arm!`;
        if (resData.plan) renderMissionOnMap(resData.plan);
      } else {
        el.alertBox.className = 'alert-box error';
        el.alertBox.textContent = `Upload failed: ${resData.message || 'Error'}`;
      }
    } catch (e) {
      el.alertBox.className = 'alert-box error';
      el.alertBox.textContent = `Network error: ${e.message}`;
    }
  });

  const sendMissionCmd = async (endpoint, payload = {}) => {
    try {
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      return await res.json();
    } catch (e) {
      console.error(e);
      return { status: 'error', message: e.message };
    }
  };

  el.btnModalStart.addEventListener('click', async () => {
    const r = await sendMissionCmd('/api/mission/start', { auto_arm: true });
    if (r.status === 'success') {
      el.alertBox.className = 'alert-box success';
      el.alertBox.textContent = '🚀 Pixhawk commanded to AUTO mode! Autonomous mission started.';
      setTimeout(closeModal, 1500);
    }
  });
  el.btnHudStart.addEventListener('click', () => sendMissionCmd('/api/mission/start', { auto_arm: true }));

  el.btnModalPause.addEventListener('click', () => sendMissionCmd('/api/mission/pause'));
  el.btnHudPause.addEventListener('click', () => sendMissionCmd('/api/mission/pause'));

  el.btnModalAbort.addEventListener('click', () => sendMissionCmd('/api/mission/abort'));
  el.btnHudRtl.addEventListener('click', () => sendMissionCmd('/api/mission/abort'));

  el.btnModalClear.addEventListener('click', async () => {
    await sendMissionCmd('/api/mission/clear');
    missionLayerGroup.clearLayers();
    activeWpMarkers = [];
    el.alertBox.className = 'alert-box';
    el.alertBox.textContent = 'Mission cleared from Pixhawk memory.';
  });

  // 8. WEBSOCKET & SSE CONNECTION
  let lastRxTimestamp = Date.now();

  function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;

    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      el.connText.textContent = 'PIXHAWK LIVE';
      el.connIndicator.className = 'conn-indicator online';
    };

    ws.onmessage = (event) => {
      try {
        lastRxTimestamp = Date.now();
        const payload = JSON.parse(event.data);
        updateTelemetry(payload);
      } catch (err) {
        console.error('JSON parse error:', err);
      }
    };

    ws.onerror = () => ws.close();

    ws.onclose = () => {
      el.connText.textContent = 'NO HEARTBEAT (>10s)';
      el.connIndicator.className = 'conn-indicator lost';
      setTimeout(connectWebSocket, 2000);
    };
  }

  setInterval(async () => {
    try {
      const res = await fetch('/api/telemetry');
      if (res.ok) {
        lastRxTimestamp = Date.now();
        const data = await res.json();
        updateTelemetry(data);
      }
    } catch (e) {}
  }, 500);

  // 10-Second Heartbeat Watchdog
  setInterval(() => {
    if (Date.now() - lastRxTimestamp > 10000) {
      el.badgeMode.textContent = 'NO HEARTBEAT';
      el.badgeMode.className = 'status-badge lost';
      el.connText.textContent = 'NO HEARTBEAT (>10s)';
      el.connIndicator.className = 'conn-indicator lost';
    }
  }, 1000);

  connectWebSocket();
});
