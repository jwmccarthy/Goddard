import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

const $ = (id) => document.getElementById(id);
const view = $('view');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
renderer.shadowMap.enabled = true;
view.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d222e);
scene.fog = new THREE.Fog(0x0d222e, 10500, 19000);
scene.add(new THREE.HemisphereLight(0xe3faf3, 0x527e82, 2.0));
const sun = new THREE.DirectionalLight(0xffebc4, 3.0);
sun.position.set(-3400, -4100, 7400);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.left = sun.shadow.camera.bottom = -7000;
sun.shadow.camera.top = sun.shadow.camera.right = 7000;
scene.add(sun);

const camera = new THREE.PerspectiveCamera(55, 1, 10, 30000);
camera.up.set(0, 0, 1);
camera.position.set(4800, -5800, 4000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 420);
controls.enableDamping = true;
controls.dampingFactor = 0.065;
controls.minDistance = 300;
controls.maxDistance = 15000;
controls.maxPolarAngle = Math.PI * 0.49;

for (const [y, color] of [[-2560, 0x1a494f], [2560, 0x1a424d]]) {
  const half = new THREE.Mesh(
    new THREE.PlaneGeometry(8192, 5120),
    new THREE.MeshStandardMaterial({ color, roughness: 0.92 }),
  );
  half.position.set(0, y, -4);
  half.receiveShadow = true;
  scene.add(half);
}

function addLine(points, color = 0x7bbfb7, opacity = 0.44) {
  const geometry = new THREE.BufferGeometry().setFromPoints(
    points.map(([x, y]) => new THREE.Vector3(x, y, 7)),
  );
  scene.add(new THREE.Line(
    geometry, new THREE.LineBasicMaterial({ color, transparent: true, opacity }),
  ));
}
addLine([[-4096, 0], [4096, 0]]);
addLine([[-4096, -5120], [4096, -5120], [4096, 5120], [-4096, 5120], [-4096, -5120]], 0xb7d7d1, 0.5);
addLine(Array.from({ length: 65 }, (_, i) => {
  const angle = i / 64 * Math.PI * 2;
  return [Math.cos(angle) * 920, Math.sin(angle) * 920];
}));

new OBJLoader().load('/arena.obj', (arena) => {
  arena.traverse((child) => {
    if (!child.isMesh) return;
    child.material = new THREE.MeshStandardMaterial({
      color: 0x83aab2, transparent: true, opacity: 0.15,
      depthWrite: false, roughness: 0.65, side: THREE.DoubleSide,
    });
  });
  arena.rotation.z = Math.PI / 2;
  scene.add(arena);
});

function makeCar(color) {
  const group = new THREE.Group();
  const material = new THREE.MeshStandardMaterial({
    color, emissive: color, emissiveIntensity: 0, metalness: .18, roughness: .35,
  });
  const body = new THREE.Mesh(new THREE.BoxGeometry(120, 87, 39), material);
  body.castShadow = true;
  body.receiveShadow = true;
  group.add(body);
  const nose = new THREE.Mesh(
    new THREE.ConeGeometry(19, 39, 3),
    new THREE.MeshStandardMaterial({ color: 0xe8f9f2 }),
  );
  nose.position.x = 74;
  nose.rotation.z = -Math.PI / 2;
  group.add(nose);
  scene.add(group);
  return { group, material };
}

const cars = [makeCar(0x5e9dfb), makeCar(0xff955b)];
const ball = new THREE.Mesh(
  new THREE.SphereGeometry(91.25, 24, 16),
  new THREE.MeshStandardMaterial({ color: 0xf8f4e9, emissive: 0x275451, roughness: .28 }),
);
ball.castShadow = true;
scene.add(ball);
const pathGroup = new THREE.Group();
scene.add(pathGroup);
const basis = new THREE.Matrix4();
const forward = new THREE.Vector3();
const right = new THREE.Vector3();
const up = new THREE.Vector3();

function setCar(index, state) {
  const car = cars[index];
  car.group.visible = !state.demoed;
  car.group.position.fromArray(state.pos);
  forward.fromArray(state.fwd);
  right.fromArray(state.rgt);
  up.fromArray(state.up);
  basis.makeBasis(forward, right, up);
  car.group.quaternion.setFromRotationMatrix(basis);
  car.material.emissiveIntensity = state.boosting ? 1.1 : 0;
}

function setPath(frames) {
  for (const item of [...pathGroup.children]) {
    pathGroup.remove(item);
    item.geometry.dispose();
    item.material.dispose();
  }
  if (!frames.length) return;
  const positions = frames.map((frame) => new THREE.Vector3(...frame.ball));
  const trace = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(positions),
    new THREE.LineBasicMaterial({ color: 0x91e8da, transparent: true, opacity: .42 }),
  );
  pathGroup.add(trace);
  for (const frame of frames) {
    if (!frame.touch) continue;
    const dot = new THREE.Mesh(
      new THREE.SphereGeometry(37, 10, 8),
      new THREE.MeshBasicMaterial({ color: 0xfac075 }),
    );
    dot.position.fromArray(frame.ball);
    pathGroup.add(dot);
  }
}

let clip = null;
let frameIndex = 0;
let playing = false;
let speedIndex = 1;
const speeds = [0.5, 1, 2, 4];
let currentSkill = 'all';
let selectedId = null;
let pageOffset = 0;
let pageTotal = 0;
let catalogRequest = 0;
let clipRequest = 0;
let readyVersion = null;
let polling = false;
const percent = (value) => `${Math.round(value * 100)}%`;
const scoreColor = (value) => value > .5 ? 'error' : 'good';

async function getJSON(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `${response.status} ${response.statusText}`);
  return data;
}

function statusMessage(text, kind = '') {
  $('statusPill').textContent = text;
  $('statusPill').className = `status-pill ${kind}`;
}

function showScan(title, text, progress = 0) {
  $('scanScreen').classList.remove('hidden');
  $('scanTitle').textContent = title;
  $('scanText').textContent = text;
  $('scanProgress').style.width = `${Math.max(5, Math.min(100, progress * 100))}%`;
}

async function refreshCheckpoints() {
  try {
    const checkpoints = await getJSON('/api/checkpoints');
    const picker = $('checkpointSelect');
    const previous = picker.value;
    picker.replaceChildren();
    for (const checkpoint of checkpoints) {
      const option = document.createElement('option');
      option.value = checkpoint.path;
      option.textContent = `${checkpoint.label} · ${Number(checkpoint.step).toLocaleString()} steps`;
      picker.appendChild(option);
    }
    if (checkpoints.some((item) => item.path === previous)) picker.value = previous;
  } catch (error) {
    statusMessage(error.message, 'error');
  }
}

function configureHeads(heads) {
  const select = $('headSelect');
  const previous = select.value;
  select.replaceChildren();
  for (const head of heads) {
    const option = document.createElement('option');
    option.value = head;
    option.textContent = head === 'combined' ? 'Combined' : `${head[0].toUpperCase()}${head.slice(1)} head`;
    select.appendChild(option);
  }
  if (heads.includes(previous)) select.value = previous;
}

function clearClip(title) {
  clip = null;
  selectedId = null;
  playing = false;
  frameIndex = 0;
  progressFrame = 0;
  setPath([]);
  cars.forEach((car) => { car.group.visible = false; });
  ball.visible = false;
  $('stageTitle').textContent = title;
  $('stageSubtitle').textContent = 'Orbit: drag · Zoom: scroll · Playback: space / arrow keys';
  $('frameSlider').max = 0;
  $('frameSlider').value = 0;
  $('frameTime').textContent = '0.00 s';
  $('frameLabel').textContent = 'No replay selected';
  $('playButton').textContent = '▶';
  $('clipTitle').textContent = 'No selection';
  $('clipSource').textContent = 'Choose a play to inspect its source and decision trace.';
  for (const id of ['clipActor', 'clipSplit', 'clipRows', 'clipDuration', 'clipCurrent',
                    'meanScore', 'missScore', 'peakScore']) $(id).textContent = '—';
  for (const id of ['meanScore', 'missScore']) $(id).classList.remove('warning');
  for (const part of $('phaseBar').children) part.style.width = '0%';
  drawChart();
}

async function pollStatus() {
  if (polling) return;
  polling = true;
  try {
    const status = await getJSON('/api/status');
    if (status.phase === 'ready') {
      $('scanScreen').classList.add('hidden');
      statusMessage(`${Number(status.step).toLocaleString()} steps · ${status.clips.toLocaleString()} clips`, 'ready');
      $('scannedCount').textContent = `${status.windows.toLocaleString()} windows`;
      const version = `${status.checkpoint}:${status.step}:${status.replay_dir}`;
      if (version !== readyVersion) {
        readyVersion = version;
        clearClip('Select a sequence');
        configureHeads(status.heads);
        await loadCatalog(true);
      }
    } else {
      readyVersion = null;
      const done = Number(status.done || 0);
      const total = Number(status.total || 0);
      if (status.phase === 'scoring') {
        const progress = total ? done / total : 0;
        statusMessage(`Scoring ${percent(progress)}`);
        showScan('Reading the discriminator…', `${done.toLocaleString()} of ${total.toLocaleString()} expert windows scored.`, progress);
      } else if (status.phase === 'error') {
        statusMessage('Scan failed', 'error');
        showScan('Could not score sequences', status.error || 'Unknown checkpoint error', 0);
      } else {
        statusMessage(status.phase === 'cataloguing' ? 'Indexing replays' : 'Loading');
        showScan('Building the expert catalog…', status.replay_dir || status.checkpoint || 'Loading checkpoint', 0);
      }
    }
  } catch (error) {
    statusMessage('Disconnected', 'error');
    showScan('Cannot reach the inspector', error.message);
  } finally {
    polling = false;
  }
}

async function scan(checkpoint) {
  try {
    await getJSON('/api/scan', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ checkpoint }),
    });
    readyVersion = null;
    clearClip('Loading checkpoint…');
    $('sequenceList').replaceChildren();
    $('resultCount').textContent = 'Scanning';
    $('loadMore').hidden = true;
    showScan('Loading checkpoint…', 'Recomputing expert scores');
    await pollStatus();
    await refreshCheckpoints();
  } catch (error) {
    statusMessage(error.message, 'error');
  }
}

function listCard(item) {
  const card = document.createElement('button');
  card.className = `sequence-item${item.id === selectedId ? ' selected' : ''}`;
  card.dataset.id = item.id;
  const top = document.createElement('div');
  top.className = 'item-top';
  const name = document.createElement('span');
  name.className = 'item-skill';
  name.textContent = `${item.skill} / ${String(item.id).padStart(4, '0')}`;
  const score = document.createElement('span');
  score.className = `item-score ${scoreColor(item.mean_agent)}`;
  score.textContent = percent(item.mean_agent);
  top.append(name, score);
  const source = document.createElement('div');
  source.className = 'item-source';
  source.title = item.source;
  source.textContent = item.source;
  const bottom = document.createElement('div');
  bottom.className = 'item-bottom';
  const split = document.createElement('span');
  split.className = `chip ${item.split}`;
  split.textContent = item.split;
  const context = document.createElement('span');
  context.textContent = `${percent(item.miss_fraction)} failed · ${item.length} frames`;
  bottom.append(split, context);
  card.append(top, source, bottom);
  card.addEventListener('click', () => selectSequence(item.id));
  return card;
}

async function loadCatalog(reset = true) {
  if (!readyVersion) return;
  const request = ++catalogRequest;
  if (reset) {
    pageOffset = 0;
    selectedId = null;
    clipRequest += 1;
    $('sequenceList').replaceChildren();
  }
  const query = new URLSearchParams({
    skill: currentSkill, split: $('splitSelect').value,
    head: $('headSelect').value, order: $('orderSelect').value,
    search: $('searchInput').value, offset: pageOffset, limit: 40,
  });
  try {
    const results = await getJSON(`/api/sequences?${query}`);
    if (request !== catalogRequest) return;
    for (const item of results.items) $('sequenceList').appendChild(listCard(item));
    pageOffset = results.offset + results.items.length;
    pageTotal = results.total;
    $('resultCount').textContent = `${pageTotal.toLocaleString()} matching clips`;
    $('loadMore').hidden = pageOffset >= pageTotal;
    if (reset && results.items.length) await selectSequence(results.items[0].id);
    if (reset && !results.items.length) {
      const empty = document.createElement('p');
      empty.className = 'note';
      empty.textContent = 'No sequences match these filters.';
      $('sequenceList').appendChild(empty);
      clearClip('No matching sequence');
    }
  } catch (error) {
    if (request === catalogRequest) statusMessage(error.message, 'error');
  }
}

function focusCamera() {
  if (!clip) return;
  const point = new THREE.Vector3(...clip.frames[Math.min(clip.action_start, clip.frames.length - 1)].ball);
  controls.target.copy(point);
  camera.position.copy(point).add(new THREE.Vector3(2600, -3400, 2500));
  controls.update();
}

async function selectSequence(id) {
  const request = ++clipRequest;
  try {
    const data = await getJSON(`/api/sequence/${id}`);
    if (request !== clipRequest || !readyVersion) return;
    clip = data;
    selectedId = id;
    playing = true;
    frameIndex = 0;
    progressFrame = 0;
    ball.visible = true;
    $('frameSlider').max = Math.max(0, clip.frames.length - 1);
    $('stageTitle').textContent = `${clip.skill[0].toUpperCase()}${clip.skill.slice(1)} · expert #${clip.id}`;
    $('stageSubtitle').textContent = `${clip.split} · ${clip.source} · ${clip.frames.length} frames`;
    for (const card of document.querySelectorAll('.sequence-item')) {
      card.classList.toggle('selected', Number(card.dataset.id) === id);
    }
    $('clipTitle').textContent = `${clip.skill} / #${clip.id}`;
    $('clipSource').textContent = clip.source;
    $('clipActor').textContent = clip.actor === 0 ? 'Blue' : 'Orange';
    $('clipSplit').textContent = clip.split === 'heldout' ? 'Held-out' : 'Training';
    $('clipRows').textContent = `${clip.source_start.toLocaleString()} – ${clip.source_stop.toLocaleString()}`;
    $('clipDuration').textContent = `${(clip.frames.length * clip.frame_skip / 120).toFixed(2)} s`;
    const setup = clip.action_start / clip.frames.length * 100;
    const action = (clip.action_stop - clip.action_start) / clip.frames.length * 100;
    const phases = $('phaseBar').children;
    phases[0].style.width = `${setup}%`;
    phases[1].style.width = `${action}%`;
    phases[2].style.width = `${Math.max(0, 100 - setup - action)}%`;
    setPath(clip.frames);
    focusCamera();
    showFrame(0);
    updateMetrics();
  } catch (error) {
    if (request === clipRequest) statusMessage(error.message, 'error');
  }
}

function activeScores() {
  if (!clip) return [];
  return clip.scores[$('headSelect').value] || clip.scores.combined;
}

function updateMetrics() {
  if (!clip) return;
  const scores = activeScores();
  const average = scores.reduce((sum, value) => sum + value, 0) / scores.length;
  const miss = scores.filter((value) => value > .5).length / scores.length;
  const peak = Math.max(...scores);
  $('meanScore').textContent = percent(average);
  $('missScore').textContent = percent(miss);
  $('peakScore').textContent = percent(peak);
  $('meanScore').classList.toggle('warning', average > .5);
  $('missScore').classList.toggle('warning', miss > .5);
  drawChart();
}

function showFrame(index) {
  if (!clip) return;
  frameIndex = Math.max(0, Math.min(clip.frames.length - 1, Math.round(index)));
  const state = clip.frames[frameIndex];
  ball.position.fromArray(state.ball);
  state.cars.forEach((car, idx) => setCar(idx, car));
  $('frameSlider').value = frameIndex;
  $('frameTime').textContent = `${(frameIndex * clip.frame_skip / 120).toFixed(2)} s`;
  $('frameLabel').textContent = `row ${clip.source_start + frameIndex} / ${clip.source_stop}${state.touch ? ' · EGO TOUCH' : ''}`;
  $('clipCurrent').textContent = `${percent(activeScores()[frameIndex])} P(agent)`;
  $('playButton').textContent = playing ? 'Ⅱ' : '▶';
  drawChart();
}

const chart = $('scoreChart');
const chartPad = { left: 34, right: 12, top: 11, bottom: 15 };

function chartPosition(clientX) {
  if (!clip) return 0;
  const bounds = chart.getBoundingClientRect();
  const usable = Math.max(1, bounds.width - chartPad.left - chartPad.right);
  const part = Math.max(0, Math.min(1, (clientX - bounds.left - chartPad.left) / usable));
  return Math.round(part * (clip.frames.length - 1));
}

function drawChart() {
  const bounds = chart.getBoundingClientRect();
  if (!bounds.width || !bounds.height) return;
  const pixelRatio = Math.min(devicePixelRatio, 2);
  const pixelWidth = Math.round(bounds.width * pixelRatio);
  const pixelHeight = Math.round(bounds.height * pixelRatio);
  if (chart.width !== pixelWidth || chart.height !== pixelHeight) {
    chart.width = pixelWidth;
    chart.height = pixelHeight;
  }
  const ctx = chart.getContext('2d');
  ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  const width = bounds.width;
  const height = bounds.height;
  ctx.clearRect(0, 0, width, height);
  const graphWidth = width - chartPad.left - chartPad.right;
  const graphHeight = height - chartPad.top - chartPad.bottom;
  ctx.font = '10px ui-monospace, monospace';
  ctx.textAlign = 'right';
  for (const value of [1, .5, 0]) {
    const y = chartPad.top + (1 - value) * graphHeight;
    ctx.strokeStyle = value === .5 ? '#ca987d' : '#355363';
    ctx.setLineDash(value === .5 ? [5, 5] : []);
    ctx.beginPath(); ctx.moveTo(chartPad.left, y); ctx.lineTo(width - chartPad.right, y); ctx.stroke();
    ctx.fillStyle = '#8ea6ae';
    ctx.fillText(`${Math.round(value * 100)}`, chartPad.left - 5, y + 3);
  }
  ctx.setLineDash([]);
  if (!clip) return;
  const scores = activeScores();
  const interval = graphWidth / Math.max(1, scores.length);
  ctx.fillStyle = '#5f94d621';
  ctx.fillRect(chartPad.left + clip.action_start * interval, chartPad.top,
    (clip.action_stop - clip.action_start) * interval, graphHeight);
  scores.forEach((value, index) => {
    ctx.fillStyle = value > .5 ? '#fba083c0' : '#7aead2a5';
    const barHeight = Math.max(2, value * graphHeight);
    ctx.fillRect(chartPad.left + index * interval, chartPad.top + graphHeight - barHeight,
      Math.max(1, interval * .86), barHeight);
    if (clip.frames[index].touch) {
      ctx.fillStyle = '#ffdb83';
      ctx.fillRect(chartPad.left + index * interval, height - 6, Math.max(2, interval), 5);
    }
  });
  ctx.strokeStyle = '#f1f8ee';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  const x = chartPad.left + (frameIndex + .5) * interval;
  ctx.moveTo(x, chartPad.top);
  ctx.lineTo(x, chartPad.top + graphHeight);
  ctx.stroke();
}

chart.addEventListener('pointermove', (event) => {
  if (!clip) return;
  const index = chartPosition(event.clientX);
  const tip = $('chartTooltip');
  tip.textContent = `row ${clip.source_start + index} · P(agent) ${percent(activeScores()[index])}`;
  tip.style.display = 'block';
  tip.style.left = `${Math.min(event.offsetX + 12, chart.clientWidth - 170)}px`;
  tip.style.top = '4px';
});
chart.addEventListener('pointerleave', () => { $('chartTooltip').style.display = 'none'; });
chart.addEventListener('pointerdown', (event) => { playing = false; showFrame(chartPosition(event.clientX)); });
$('frameSlider').addEventListener('input', (event) => { playing = false; showFrame(Number(event.target.value)); });
$('playButton').addEventListener('click', () => {
  if (!clip) return;
  playing = !playing;
  $('playButton').textContent = playing ? 'Ⅱ' : '▶';
});
$('speedButton').addEventListener('click', () => {
  speedIndex = (speedIndex + 1) % speeds.length;
  $('speedButton').textContent = `${speeds[speedIndex]}× speed`;
});
document.addEventListener('keydown', (event) => {
  if (!clip || ['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
  if (event.code === 'Space') {
    event.preventDefault();
    $('playButton').click();
  } else if (event.code === 'ArrowRight' || event.code === 'ArrowLeft') {
    event.preventDefault();
    playing = false;
    showFrame(frameIndex + (event.code === 'ArrowRight' ? 1 : -1));
  } else if (event.key.toLowerCase() === 'r') {
    focusCamera();
  }
});

for (const button of document.querySelectorAll('.skill-button')) {
  button.addEventListener('click', () => {
    currentSkill = button.dataset.skill;
    document.querySelector('.skill-button.active')?.classList.remove('active');
    button.classList.add('active');
    loadCatalog(true);
  });
}
for (const id of ['splitSelect', 'headSelect', 'orderSelect']) {
  $(id).addEventListener('change', () => {
    if (id === 'headSelect') updateMetrics();
    loadCatalog(true);
  });
}
let searchTimer;
$('searchInput').addEventListener('input', () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => loadCatalog(true), 250);
});
$('loadMore').addEventListener('click', () => loadCatalog(false));
$('scanSelected').addEventListener('click', () => scan($('checkpointSelect').value));
$('scanLatest').addEventListener('click', () => scan('latest'));

const resize = () => {
  const width = view.clientWidth;
  const height = view.clientHeight;
  if (!width || !height) return;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
  drawChart();
};
new ResizeObserver(resize).observe(view);
new ResizeObserver(drawChart).observe(chart);
resize();
let previousTime = performance.now();
let progressFrame = 0;
function render(time) {
  const delta = Math.min(.1, (time - previousTime) / 1000);
  previousTime = time;
  if (playing && clip) {
    progressFrame += delta * 120 / clip.frame_skip * speeds[speedIndex];
    if (progressFrame >= 1) {
      const steps = Math.floor(progressFrame);
      progressFrame -= steps;
      showFrame((frameIndex + steps) % clip.frames.length);
    }
  } else {
    progressFrame = 0;
  }
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(render);
}
requestAnimationFrame(render);
clearClip('Select a sequence');
pollStatus();
refreshCheckpoints();
setInterval(pollStatus, 1000);
setInterval(refreshCheckpoints, 10000);
