from __future__ import annotations

import json
import math
import threading
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fitness_evaluator import FitnessEvaluator
from network_parser import parse_inp_file
from persistence import RunPersistence


class BrowserReplayExporter:
    def __init__(self, base_dir: Path, persistence: RunPersistence):
        self.base_dir = Path(base_dir)
        self.persistence = persistence
        self.data_dir = self.base_dir / "data"
        self.output_dir = self.base_dir / "Memetic_GA" / "Attempt_005" / "results" / "browser_replays"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._http_server: Optional[ThreadingHTTPServer] = None
        self._http_thread: Optional[threading.Thread] = None
        self._http_port: Optional[int] = None

    def export_run(self, run_id: str) -> Path:
        run_row = self.persistence.load_run(run_id)
        if not run_row:
            raise ValueError(f"Run {run_id} was not found")

        network_file = str(run_row.get("network_file", ""))
        network_path = self.data_dir / network_file
        network = parse_inp_file(str(network_path))
        generations = self.persistence.load_generations(run_id)
        generations = [g for g in generations if g.get("best_chromosome") is not None]
        if not generations:
            raise ValueError("This run does not have stored chromosome snapshots yet")

        diameter_values, unit_cost_lookup = self._get_benchmark_cost_spec(network_file)
        evaluator = FitnessEvaluator(
            network,
            diameter_options=diameter_values,
            unit_cost_lookup=unit_cost_lookup if unit_cost_lookup else None,
        )

        nodes, pipes = self._load_layout(network_path)
        if not nodes:
            raise ValueError("Could not load node coordinates for browser replay")

        node_values = {node_id: float(node["elevation"]) for node_id, node in nodes.items() if node.get("elevation") is not None}
        node_positions = {node_id: [float(node["x"]), float(node["y"])] for node_id, node in nodes.items()}

        frames = []
        for row in generations:
            chromosome = [int(g) for g in (row.get("best_chromosome") or [])]
            diameters = evaluator.indices_to_diameters(chromosome)
            frames.append(
                {
                    "generation": int(row["generation"]),
                    "training_fitness": self._to_float(row.get("best_training_fitness")),
                    "paper_score": self._to_float(row.get("best_paper_score")),
                    "paper_cost": self._to_float(row.get("best_paper_cost")),
                    "feasible_count": int(row.get("feasible_count") or 0),
                    "gap_to_published_pct": self._to_float(row.get("gap_to_published_pct")),
                    "chromosome": chromosome,
                    "diameters": diameters,
                }
            )

        payload = {
            "run_id": run_id,
            "network_file": network_file,
          "population_size": int((run_row.get("config") or {}).get("population_size") or 0),
            "network": {
                "nodes": nodes,
                "pipes": pipes,
                "node_positions": node_positions,
                "node_values": node_values,
                "diameter_values": [float(d) for d in diameter_values],
            },
            "frames": frames,
        }

        output_path = self.output_dir / f"{run_id}.html"
        output_path.write_text(self._build_html(payload), encoding="utf-8")
        return output_path

    def export_and_open(self, run_id: str) -> Path:
        output_path = self.export_run(run_id)
        port = self._ensure_http_server()
        webbrowser.open_new_tab(f"http://127.0.0.1:{port}/{output_path.name}")
        return output_path

    def _ensure_http_server(self) -> int:
        if self._http_server is not None and self._http_port is not None:
            return self._http_port

        handler = partial(SimpleHTTPRequestHandler, directory=str(self.output_dir))
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        self._http_server = server
        self._http_thread = thread
        self._http_port = int(server.server_address[1])
        return self._http_port

    def shutdown(self) -> None:
        if self._http_server is None:
            return
        try:
            self._http_server.shutdown()
            self._http_server.server_close()
        except Exception:
            pass
        self._http_server = None
        self._http_thread = None
        self._http_port = None

    def _get_benchmark_cost_spec(self, network_file: str) -> Tuple[List[float], Dict[float, float]]:
        try:
            from test_benchmarks import BenchmarkRunner

            results_dir = self.base_dir / "Memetic_GA" / "Attempt_004" / "results"
            runner = BenchmarkRunner(str(self.data_dir), str(results_dir))
            return runner._get_benchmark_cost_spec(network_file)
        except Exception:
            return [], {}

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    def _load_layout(self, network_path: Path) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
        try:
            import wntr

            wn = wntr.network.WaterNetworkModel(str(network_path))
            nodes: Dict[str, Dict[str, Any]] = {}
            coords_by_node: Dict[str, Tuple[float, float]] = {}

            for name in wn.node_name_list:
                node = wn.get_node(name)
                nodes[name] = {
                    "elevation": self._to_float(getattr(node, "elevation", None)),
                    "kind": "reservoir" if name in wn.reservoir_name_list else "junction",
                }
                coords = getattr(node, "coordinates", None)
                if coords and len(coords) >= 2:
                  coords_by_node[name] = (float(coords[0]), float(coords[1]))

            pipes = []
            for link_name in wn.pipe_name_list:
                link = wn.get_link(link_name)
                pipes.append(
                    {
                        "id": str(link_name),
                        "node1": str(link.start_node_name),
                        "node2": str(link.end_node_name),
                    "length_m": self._to_float(getattr(link, "length", None)),
                    }
                )

            for pipe in pipes:
                nodes.setdefault(pipe["node1"], {"elevation": None, "kind": "junction"})
                nodes.setdefault(pipe["node2"], {"elevation": None, "kind": "junction"})

            missing = [node_id for node_id in nodes if node_id not in coords_by_node]
            if missing:
                placed = list(coords_by_node.values())
                if placed:
                    cx = sum(x for x, _ in placed) / len(placed)
                    cy = sum(y for _, y in placed) / len(placed)
                    span_x = max(x for x, _ in placed) - min(x for x, _ in placed)
                    span_y = max(y for _, y in placed) - min(y for _, y in placed)
                    radius = max(span_x, span_y, 1.0) * 0.6
                else:
                    cx, cy, radius = 0.0, 0.0, float(max(len(nodes), 1))

                step = 2.0 * math.pi / max(len(missing), 1)
                for idx, node_id in enumerate(sorted(missing)):
                    angle = idx * step
                    coords_by_node[node_id] = (
                        cx + radius * math.cos(angle),
                        cy + radius * math.sin(angle),
                    )

            for node_id, meta in nodes.items():
                x, y = coords_by_node[node_id]
                meta["x"] = float(x)
                meta["y"] = float(y)

            return nodes, pipes
        except Exception:
            return {}, []

    def _build_html(self, payload: Dict[str, Any]) -> str:
        json_payload = json.dumps(payload, ensure_ascii=False)
        title = escape(f"Solution Replay - {payload['network_file']} - {payload['run_id']}")

        html = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>__TITLE__</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f6f8;
      --panel: #ffffff;
      --ink: #14202b;
      --muted: #5a6874;
      --accent: #0f766e;
      --border: #d8e0e8;
      --shadow: 0 14px 40px rgba(15, 23, 42, 0.10);
    }
    html, body { height: 100%; margin: 0; }
    body {
      font-family: Segoe UI, Inter, system-ui, sans-serif;
      background: linear-gradient(180deg, #eef3f7 0%, #f8fafc 100%);
      color: var(--ink);
    }
    .shell { display: grid; grid-template-rows: auto auto 1fr; height: 100%; gap: 12px; padding: 16px; box-sizing: border-box; }
    .header, .controls, .content { background: var(--panel); border: 1px solid var(--border); border-radius: 18px; box-shadow: var(--shadow); }
    .header { padding: 16px 18px; }
    .title { font-size: 18px; font-weight: 700; margin-bottom: 6px; }
    .subtitle { color: var(--muted); font-size: 13px; line-height: 1.5; }
    .controls { padding: 14px 18px; display: grid; grid-template-columns: 1fr auto; gap: 10px 18px; align-items: center; }
    .sliderRow { display: flex; align-items: center; gap: 12px; }
    .playbackRow { margin-top: 10px; display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
    .playBtn { border: 1px solid var(--border); background: #f8fafc; color: var(--ink); border-radius: 9px; padding: 4px 10px; font-size: 12px; cursor: pointer; }
    .playBtn:hover { background: #eef2f7; }
    .speedBadge { padding: 3px 8px; border-radius: 999px; background: #e6fffb; color: #115e59; font-size: 12px; min-width: 84px; text-align: center; }
    .styleRows { margin-top: 10px; display: grid; grid-template-columns: 1fr 1fr; gap: 8px 14px; }
    .styleControl { display: grid; grid-template-columns: auto 1fr auto; align-items: center; gap: 8px; color: var(--muted); font-size: 12px; }
    .styleControl span:last-child { color: var(--ink); font-weight: 700; min-width: 44px; text-align: right; }
    input[type=range] { width: 100%; accent-color: var(--accent); }
    .metrics { display: flex; flex-wrap: wrap; gap: 10px 16px; color: var(--muted); font-size: 13px; }
    .metric strong { color: var(--ink); font-weight: 700; }
    .content { display: grid; grid-template-columns: minmax(0, 1fr) 320px; overflow: hidden; }
    .canvasWrap { position: relative; min-height: 0; padding: 10px; }
    svg { width: 100%; height: 100%; min-height: 0; display: block; background: radial-gradient(circle at top left, rgba(15, 118, 110, 0.05), transparent 40%); }
    .hoverTip { position: fixed; pointer-events: none; z-index: 20; display: none; background: rgba(15,23,42,0.92); color: #f8fafc; border: 1px solid rgba(148,163,184,0.45); border-radius: 8px; padding: 8px 10px; font-size: 12px; line-height: 1.45; white-space: pre; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
    .sidebar { border-left: 1px solid var(--border); padding: 16px; overflow: auto; background: linear-gradient(180deg, rgba(248,250,252,0.95), #fff); }
    .panelTitle { font-size: 14px; font-weight: 700; margin-bottom: 10px; }
    .mono { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; white-space: pre-wrap; font-size: 12px; line-height: 1.5; color: #21313f; }
    .legend { margin-top: 18px; padding-top: 14px; border-top: 1px solid var(--border); color: var(--muted); font-size: 12px; line-height: 1.6; }
    .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; background: #e6fffb; color: #115e59; font-size: 12px; margin-left: 8px; }
  </style>
</head>
<body>
  <div class=\"shell\">
    <div class=\"header\">
      <div class=\"title\" id=\"pageTitle\"></div>
      <div class=\"subtitle\">Fast browser replay. Slider moves through stored best solutions per generation. Pipes are colored by diameter; nodes are colored by elevation.</div>
    </div>
    <div class=\"controls\">
      <div>
        <div class=\"sliderRow\">
          <input id=\"genSlider\" type=\"range\" min=\"0\" max=\"0\" value=\"0\" step=\"1\" />
          <span class=\"pill\" id=\"genLabel\">Gen 0</span>
        </div>
        <div class="playbackRow">
          <button id="prevBtn" class="playBtn" type="button">Prev Gen</button>
          <button id="playBtn" class="playBtn" type="button">Resume</button>
          <button id="nextBtn" class="playBtn" type="button">Next Gen</button>
          <button id="speedDownBtn" class="playBtn" type="button">< Speed</button>
          <span id="speedValue" class="speedBadge">1.00 gen/s</span>
          <button id="speedUpBtn" class="playBtn" type="button">Speed ></button>
        </div>
        <div class="styleRows">
          <label class="styleControl">Pipes
            <input id="pipeWidthSlider" type="range" min="0.6" max="24.0" value="1.8" step="0.1" />
            <span id="pipeWidthValue">1.80x</span>
          </label>
          <label class="styleControl">Nodes
            <input id="nodeSizeSlider" type="range" min="0.6" max="21.0" value="1.7" step="0.1" />
            <span id="nodeSizeValue">1.70x</span>
          </label>
        </div>
      </div>
      <div class=\"metrics\">
        <div class=\"metric\"><strong id=\"metricTrain\">-</strong> train_fit</div>
        <div class=\"metric\"><strong id=\"metricPaper\">-</strong> paper_score</div>
        <div class=\"metric\"><strong id=\"metricGap\">-</strong> gap</div>
        <div class=\"metric\"><strong id=\"metricFeas\">-</strong> feasible</div>
      </div>
    </div>
    <div class=\"content\">
      <div class=\"canvasWrap\">
        <svg id=\"svg\" viewBox=\"0 0 1000 1000\" preserveAspectRatio=\"xMidYMid meet\"></svg>
      </div>
      <div class=\"sidebar\">
        <div class=\"panelTitle\">Run Info</div>
        <div class=\"mono\" id=\"infoBox\"></div>
        <div class=\"legend\">
          <div><strong>Pipe color</strong>: diameter</div>
          <div><strong>Node color</strong>: elevation</div>
          <div><strong>Node size</strong>: reservoirs emphasized</div>
          <div><strong>Controls</strong>: slider / arrow keys</div>
        </div>
      </div>
    </div>
  </div>
  <div id="hoverTip" class="hoverTip"></div>
<script>
const DATA = __JSON__;
const frames = DATA.frames;
const nodesById = DATA.network.nodes;
const pipes = DATA.network.pipes;
const nodePositions = DATA.network.node_positions;
const nodeValues = DATA.network.node_values;
const populationSize = Number(DATA.population_size || 0);
const diaMin = Math.min(...DATA.network.diameter_values);
const diaMax = Math.max(...DATA.network.diameter_values);
const nodeValsArr = Object.values(nodeValues).map(Number);
const nodeMin = nodeValsArr.length ? Math.min(...nodeValsArr) : 0;
const nodeMax = nodeValsArr.length ? Math.max(...nodeValsArr) : 1;

const slider = document.getElementById('genSlider');
const genLabel = document.getElementById('genLabel');
const infoBox = document.getElementById('infoBox');
const pageTitle = document.getElementById('pageTitle');
const metricTrain = document.getElementById('metricTrain');
const metricPaper = document.getElementById('metricPaper');
const metricGap = document.getElementById('metricGap');
const metricFeas = document.getElementById('metricFeas');
const pipeWidthSlider = document.getElementById('pipeWidthSlider');
const nodeSizeSlider = document.getElementById('nodeSizeSlider');
const pipeWidthValue = document.getElementById('pipeWidthValue');
const nodeSizeValue = document.getElementById('nodeSizeValue');
const prevBtn = document.getElementById('prevBtn');
const nextBtn = document.getElementById('nextBtn');
const playBtn = document.getElementById('playBtn');
const speedDownBtn = document.getElementById('speedDownBtn');
const speedUpBtn = document.getElementById('speedUpBtn');
const speedValue = document.getElementById('speedValue');
const hoverTip = document.getElementById('hoverTip');
const svg = document.getElementById('svg');

pageTitle.textContent = `${DATA.network_file} replay for ${DATA.run_id}`;
slider.max = String(frames.length - 1);
slider.value = '0';

const NS = 'http://www.w3.org/2000/svg';
const elements = { pipes: [], nodes: [], labels: [] };
let currentFrameIndex = 0;
let pipeWidthScale = Number(pipeWidthSlider.value || 1.8);
let nodeSizeScale = Number(nodeSizeSlider.value || 1.7);
let playbackSpeed = 1.0;
let playTimer = null;
const nodeDegree = {};
pipes.forEach((p) => {
  nodeDegree[p.node1] = (nodeDegree[p.node1] || 0) + 1;
  nodeDegree[p.node2] = (nodeDegree[p.node2] || 0) + 1;
});

function updateStyleLabels() {
  pipeWidthValue.textContent = `${pipeWidthScale.toFixed(2)}x`;
  nodeSizeValue.textContent = `${nodeSizeScale.toFixed(2)}x`;
}

function updatePlaybackLabel() {
  speedValue.textContent = `${playbackSpeed.toFixed(2)} gen/s`;
  playBtn.textContent = playTimer ? 'Pause' : 'Resume';
}

function stepGeneration(delta) {
  const cur = parseInt(slider.value, 10) || 0;
  const nxt = Math.max(0, Math.min(frames.length - 1, cur + delta));
  slider.value = String(nxt);
  scheduleRender(nxt);
}

function stopPlayback() {
  if (playTimer) {
    clearInterval(playTimer);
    playTimer = null;
  }
  updatePlaybackLabel();
}

function startPlayback() {
  if (playTimer || frames.length <= 1) {
    updatePlaybackLabel();
    return;
  }
  playTimer = setInterval(() => {
    const cur = parseInt(slider.value, 10) || 0;
    if (cur >= frames.length - 1) {
      stopPlayback();
      return;
    }
    stepGeneration(1);
  }, Math.max(40, Math.round(1000 / Math.max(0.05, playbackSpeed))));
  updatePlaybackLabel();
}

function togglePlayback() {
  if (playTimer) {
    stopPlayback();
  } else {
    startPlayback();
  }
}

function changeSpeed(multiplier) {
  playbackSpeed = Math.max(0.1, Math.min(20.0, playbackSpeed * multiplier));
  const wasPlaying = Boolean(playTimer);
  if (wasPlaying) {
    stopPlayback();
    startPlayback();
  }
  updatePlaybackLabel();
}

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function lerp(a, b, t) { return a + (b - a) * t; }
function hex(n) { return Math.round(clamp(n, 0, 255)).toString(16).padStart(2, '0'); }
function mixColor(c1, c2, t) {
  const p1 = [parseInt(c1.slice(1,3),16), parseInt(c1.slice(3,5),16), parseInt(c1.slice(5,7),16)];
  const p2 = [parseInt(c2.slice(1,3),16), parseInt(c2.slice(3,5),16), parseInt(c2.slice(5,7),16)];
  return '#' + hex(lerp(p1[0], p2[0], t)) + hex(lerp(p1[1], p2[1], t)) + hex(lerp(p1[2], p2[2], t));
}
const COLOR_GREEN = '#16a34a';
const COLOR_RED = '#dc2626';
function diameterColor(d) {
  const t = (d - diaMin) / (diaMax - diaMin || 1);
  return mixColor(COLOR_GREEN, COLOR_RED, t);
}
function elevationColor(v) {
  const t = (v - nodeMin) / (nodeMax - nodeMin || 1);
  return mixColor(COLOR_GREEN, COLOR_RED, t);
}

function bounds() {
  const xs = Object.values(nodePositions).map(p => p[0]);
  const ys = Object.values(nodePositions).map(p => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const padX = (maxX - minX) * 0.07 || 10;
  const padY = (maxY - minY) * 0.07 || 10;
  return { x: minX - padX, y: minY - padY, w: (maxX - minX) + 2*padX, h: (maxY - minY) + 2*padY };
}

function buildSvg() {
  const b = bounds();
  svg.setAttribute('viewBox', `${b.x} ${b.y} ${b.w} ${b.h}`);

  pipes.forEach((p, idx) => {
    const line = document.createElementNS(NS, 'line');
    const a = nodePositions[p.node1];
    const b = nodePositions[p.node2];
    if (!a || !b) return;
    line.setAttribute('x1', a[0]); line.setAttribute('y1', a[1]);
    line.setAttribute('x2', b[0]); line.setAttribute('y2', b[1]);
    line.setAttribute('stroke', '#475569');
    line.setAttribute('stroke-width', '3.0');
    line.setAttribute('opacity', '1');
    line.setAttribute('stroke-linecap', 'round');
    line.addEventListener('mouseenter', () => {
      const frame = frames[currentFrameIndex] || {};
      const ds = frame.diameters || [];
      const d = Number(ds[idx]);
      const diameterText = Number.isFinite(d) ? `${d.toFixed(4)} m` : 'n/a';
      const lengthValue = Number(p.length_m);
      const lengthText = Number.isFinite(lengthValue) ? `${lengthValue.toFixed(2)} m` : 'n/a';
      hoverTip.textContent = [
        `Pipe ${p.id}`,
        `${p.node1} -> ${p.node2}`,
        `Diameter: ${diameterText}`,
        `Length: ${lengthText}`,
      ].join('\\n');
      hoverTip.style.display = 'block';
    });
    line.addEventListener('mousemove', (ev) => {
      hoverTip.style.left = `${ev.clientX + 14}px`;
      hoverTip.style.top = `${ev.clientY + 14}px`;
    });
    line.addEventListener('mouseleave', () => {
      hoverTip.style.display = 'none';
    });
    svg.appendChild(line);
    elements.pipes.push({ el: line, data: p });
  });

  Object.entries(nodePositions).forEach(([id, pos]) => {
    const circle = document.createElementNS(NS, 'circle');
    circle.setAttribute('cx', pos[0]);
    circle.setAttribute('cy', pos[1]);
    const nodeInfo = nodesById[id] || {};
    const baseR = nodeInfo.kind === 'reservoir' ? 9.0 : 4.5;
    circle.setAttribute('r', String(baseR * nodeSizeScale));
    circle.setAttribute('stroke', '#0f172a');
    circle.setAttribute('stroke-width', String(0.75 * Math.max(0.85, Math.sqrt(nodeSizeScale))));
    circle.addEventListener('mouseenter', () => {
      const nodeInfo = nodesById[id] || {};
      const kind = String(nodeInfo.kind || 'junction');
      const elevation = Number(nodeInfo.elevation);
      const elevationText = Number.isFinite(elevation) ? `${elevation.toFixed(2)} m` : 'n/a';
      const degree = Number(nodeDegree[id] || 0);
      hoverTip.textContent = [
        `Node ${id}`,
        `Type: ${kind}`,
        `Elevation: ${elevationText}`,
        `Connected pipes: ${degree}`,
      ].join('\\n');
      hoverTip.style.display = 'block';
    });
    circle.addEventListener('mousemove', (ev) => {
      hoverTip.style.left = `${ev.clientX + 14}px`;
      hoverTip.style.top = `${ev.clientY + 14}px`;
    });
    circle.addEventListener('mouseleave', () => {
      hoverTip.style.display = 'none';
    });
    svg.appendChild(circle);
    elements.nodes.push({ id, el: circle, baseR });
  });
}

function renderFrame(idx) {
  const frame = frames[idx];
  if (!frame) return;
  currentFrameIndex = idx;

  genLabel.textContent = `Gen ${idx + 1}/${frames.length}`;
  metricTrain.textContent = frame.training_fitness == null ? '-' : frame.training_fitness.toExponential(3);
  metricPaper.textContent = frame.paper_score == null ? '-' : frame.paper_score.toExponential(3);
  metricGap.textContent = frame.gap_to_published_pct == null ? 'n/a' : `${frame.gap_to_published_pct >= 0 ? '+' : ''}${frame.gap_to_published_pct.toFixed(2)}%`;
  metricFeas.textContent = populationSize > 0 ? (frame.feasible_count + ' / ' + populationSize) : String(frame.feasible_count);
  infoBox.textContent = [
    `Run: ${DATA.run_id}`,
    `Network: ${DATA.network_file}`,
    `Generation: ${frame.generation}`,
    `Training fitness: ${frame.training_fitness == null ? 'n/a' : frame.training_fitness.toExponential(3)}`,
    `Paper score: ${frame.paper_score == null ? 'n/a' : frame.paper_score.toExponential(3)}`,
    `Paper cost: ${frame.paper_cost == null ? 'n/a' : frame.paper_cost.toExponential(3)}`,
    `Feasible individuals: ${populationSize > 0 ? (frame.feasible_count + '/' + populationSize) : frame.feasible_count}`,
    `Gap to SOTA: ${frame.gap_to_published_pct == null ? 'n/a' : `${frame.gap_to_published_pct >= 0 ? '+' : ''}${frame.gap_to_published_pct.toFixed(2)}%`}`,
    '',
    'This view is browser-rendered SVG for low-lag scrubbing.'
  ].join('\\n');

  const genes = frame.diameters;
  elements.pipes.forEach((pipeRef, i) => {
    const p = pipeRef.el;
    const d = Number(genes[i]);
    if (!Number.isFinite(d)) {
      p.setAttribute('stroke', '#475569');
      p.setAttribute('stroke-width', String(3.0 * pipeWidthScale));
      p.setAttribute('opacity', '1');
      return;
    }
    const t = (d - diaMin) / (diaMax - diaMin || 1);
    p.setAttribute('stroke', diameterColor(d));
    p.setAttribute('stroke-width', String((3.0 + 5.0 * t) * pipeWidthScale));
    p.setAttribute('opacity', '1');
  });

  elements.nodes.forEach(n => {
    const value = nodeValues[n.id];
    n.el.setAttribute('r', String(n.baseR * nodeSizeScale));
    n.el.setAttribute('stroke-width', String(0.75 * Math.max(0.85, Math.sqrt(nodeSizeScale))));
    if (value == null) {
      n.el.setAttribute('fill', '#9ca3af');
      return;
    }
    n.el.setAttribute('fill', elevationColor(value));
  });
}

let rafToken = 0;
function scheduleRender(idx) {
  cancelAnimationFrame(rafToken);
  rafToken = requestAnimationFrame(() => renderFrame(idx));
}

slider.addEventListener('input', (ev) => scheduleRender(parseInt(ev.target.value, 10)));
prevBtn.addEventListener('click', () => stepGeneration(-1));
nextBtn.addEventListener('click', () => stepGeneration(1));
playBtn.addEventListener('click', () => togglePlayback());
speedDownBtn.addEventListener('click', () => changeSpeed(0.5));
speedUpBtn.addEventListener('click', () => changeSpeed(2.0));
pipeWidthSlider.addEventListener('input', (ev) => {
  pipeWidthScale = Number(ev.target.value || 1.8);
  updateStyleLabels();
  scheduleRender(parseInt(slider.value, 10));
});
nodeSizeSlider.addEventListener('input', (ev) => {
  nodeSizeScale = Number(ev.target.value || 1.7);
  updateStyleLabels();
  scheduleRender(parseInt(slider.value, 10));
});
document.addEventListener('keydown', (ev) => {
  if (ev.key === 'ArrowLeft') {
    slider.value = String(Math.max(0, parseInt(slider.value, 10) - 1));
    scheduleRender(parseInt(slider.value, 10));
  } else if (ev.key === 'ArrowRight') {
    slider.value = String(Math.min(frames.length - 1, parseInt(slider.value, 10) + 1));
    scheduleRender(parseInt(slider.value, 10));
  } else if (ev.key === ' ') {
    ev.preventDefault();
    togglePlayback();
  } else if (ev.key === 'ArrowUp') {
    changeSpeed(2.0);
  } else if (ev.key === 'ArrowDown') {
    changeSpeed(0.5);
  }
});

updateStyleLabels();
updatePlaybackLabel();
buildSvg();
renderFrame(0);
</script>
</body>
</html>"""
        return html.replace("__TITLE__", title).replace("__JSON__", json_payload)
