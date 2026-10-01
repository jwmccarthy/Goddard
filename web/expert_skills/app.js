import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

const $ = (id) => document.getElementById(id);
const SCALE = [4108, 6000, 2076];
const BLUE = 0x145bd7;
const ORANGE = 0xda572f;
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.shadowMap.enabled = true;
$('view').appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xb9cfd1);
scene.add(new THREE.HemisphereLight(0xf6fafb, 0x607d78, 2));
const sun = new THREE.DirectionalLight(0xfff4da, 3);
sun.position.set(-3500, -4500, 9000);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.left = sun.shadow.camera.bottom = -7000;
sun.shadow.camera.right = sun.shadow.camera.top = 7000;
scene.add(sun);

const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 10, 30000);
camera.up.set(0, 0, 1);
camera.position.set(5500, -6800, 4700);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 500);
controls.enableDamping = true;
controls.maxDistance = 18000;
controls.minDistance = 1200;

for (const [y, color] of [[-2560, 0x80aaa8], [2560, 0xaaa383]]) {
  const half = new THREE.Mesh(new THREE.PlaneGeometry(8192, 5120),
    new THREE.MeshStandardMaterial({ color, roughness: 0.9 }));
  half.position.set(0, y, -4);
  half.receiveShadow = true;
  scene.add(half);
}
function fieldLine(points, color, opacity = 0.7) {
  const geometry = new THREE.BufferGeometry().setFromPoints(
    points.map(([x, y]) => new THREE.Vector3(x, y, 8)));
  scene.add(new THREE.Line(geometry,
    new THREE.LineBasicMaterial({ color, transparent: true, opacity })));
}
fieldLine([[-4096, 0], [4096, 0]], 0xffffff);
fieldLine([[-4096, -5120], [4096, -5120], [4096, 5120], [-4096, 5120], [-4096, -5120]], 0xffffff);
fieldLine(Array.from({ length: 65 }, (_, i) => [
  Math.cos(i / 64 * Math.PI * 2) * 920, Math.sin(i / 64 * Math.PI * 2) * 920,
]), 0xffffff);
new OBJLoader().load('/arena.obj', (arena) => {
  arena.traverse((child) => {
    if (!child.isMesh) return;
    child.material = new THREE.MeshStandardMaterial({ color: 0x7897a0,
      transparent: true, opacity: 0.2, depthWrite: false, side: THREE.DoubleSide });
  });
  arena.rotation.z = Math.PI / 2;
  scene.add(arena);
});

function makeCar(color) {
  const group = new THREE.Group();
  const material = new THREE.MeshStandardMaterial({ color, roughness: 0.36 });
  const body = new THREE.Mesh(new THREE.BoxGeometry(120, 87, 39), material);
  body.castShadow = true;
  group.add(body);
  const nose = new THREE.Mesh(new THREE.ConeGeometry(17, 38, 3),
    new THREE.MeshStandardMaterial({ color: 0xffffff }));
  nose.rotation.z = -Math.PI / 2;
  nose.position.x = 75;
  group.add(nose);
  scene.add(group);
  return { group, material };
}
const cars = [makeCar(BLUE), makeCar(ORANGE)];
const ball = new THREE.Mesh(new THREE.SphereGeometry(91.25, 24, 16),
  new THREE.MeshStandardMaterial({ color: 0xffffff }));
ball.castShadow = true;
scene.add(ball);
const trails = new THREE.Group();
scene.add(trails);
const basis = new THREE.Matrix4();

function position(values, offset) {
  return new THREE.Vector3(...SCALE.map((factor, axis) => values[offset + axis] * factor));
}
function showScene(values) {
  ball.position.copy(position(values, 0));
  cars.forEach(({ group, material }, index) => {
    const start = 9 + index * 21;
    const forward = new THREE.Vector3(...values.slice(start + 9, start + 12));
    const up = new THREE.Vector3(...values.slice(start + 12, start + 15));
    const right = new THREE.Vector3().crossVectors(up, forward);
    group.position.copy(position(values, start)).addScaledVector(forward, 13.8757)
      .addScaledVector(up, 20.755);
    basis.makeBasis(forward, right, up);
    group.quaternion.setFromRotationMatrix(basis);
    group.visible = !values[start + 17];
    material.emissive.setHex(index ? ORANGE : BLUE);
    material.emissiveIntensity = values[start + 20] ? 0.6 : 0;
  });
}
function showTrails(frames) {
  trails.children.forEach((line) => { line.geometry.dispose(); line.material.dispose(); });
  trails.clear();
  for (const [offset, color] of [[0, 0xffffff], [9, BLUE], [30, ORANGE]]) {
    const points = frames.map((row) => position(row, offset));
    trails.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.85 })));
  }
}

let skill = null;
let allSkills = [];
let index = 0;
let playing = false;
let lastTime = performance.now();
let accumulated = 0;
let loading = 0;
let outputs = [];
let downloadUrl = null;
let prefixOutputs = [];

function format(value) {
  if (value === null || value === undefined) return '—';
  if (value === 0 || value === 1) return String(value);
  return Number(value).toFixed(4);
}
function addStat(parent, label, value) {
  const box = document.createElement('div');
  box.className = 'stat';
  const title = document.createElement('small');
  title.textContent = label;
  const content = document.createElement('strong');
  content.textContent = value;
  box.append(title, content);
  parent.appendChild(box);
}
function addFeatureGroup(parent, group, source, open) {
  const section = document.createElement('details');
  section.open = open;
  const summary = document.createElement('summary');
  summary.textContent = `${group.label} · ${group.names.length} components`;
  section.appendChild(summary);
  const body = document.createElement('div');
  body.className = 'fields';
  group.names.forEach((name, offset) => {
    const row = document.createElement('div');
    row.className = 'field';
    const title = document.createElement('span');
    title.textContent = `${group.start + offset}  ${name}`;
    const output = document.createElement('output');
    row.append(title, output);
    body.appendChild(row);
    outputs.push({ output, source, column: group.start + offset });
  });
  section.appendChild(body);
  parent.appendChild(section);
}

function inspect(selected) {
  const panel = $('inspector');
  panel.replaceChildren();
  outputs = [];
  prefixOutputs = [];
  const heading = document.createElement('h2');
  heading.textContent = `Skill ${selected.id + 1} · expert segment`;
  panel.appendChild(heading);
  const overview = document.createElement('div');
  overview.className = 'overview';
  addStat(overview, 'Sample / segment', `${selected.chunk + 1} / ${selected.segment + 1}`);
  addStat(overview, 'Source frames', `${selected.start}–${selected.stop}`);
  addStat(overview, 'Duration', `${selected.duration} steps · ${(selected.duration * selected.frameskip / 120).toFixed(2)} s`);
  addStat(overview, 'Checkpoint', selected.checkpoint);
  panel.appendChild(overview);

  selected.latents.forEach((latent, car) => {
    const box = document.createElement('div');
    box.className = car ? 'latent orange' : 'latent';
    const header = document.createElement('header');
    const name = document.createElement('span');
    name.textContent = car ? 'Orange · requested embedding' : 'Blue · requested embedding';
    const kappa = document.createElement('span');
    kappa.textContent = `κ ${selected.concentrations[car].toFixed(3)}`;
    header.append(name, kappa);
    const vector = document.createElement('pre');
    vector.textContent = `[${latent.map((value) => value.toFixed(4)).join(', ')}]`;
    box.append(header, vector);
    panel.appendChild(box);
  });
  const prefixTitle = document.createElement('h2');
  prefixTitle.textContent = 'EMA embedding of the selected expert prefix';
  prefixTitle.style.marginTop = '18px';
  panel.appendChild(prefixTitle);
  for (const car of [0, 1]) {
    const box = document.createElement('div');
    box.className = car ? 'latent orange' : 'latent';
    const header = document.createElement('header');
    const name = document.createElement('span');
    name.textContent = car ? 'Orange · current prefix' : 'Blue · current prefix';
    const kappa = document.createElement('span');
    const vector = document.createElement('pre');
    header.append(name, kappa);
    box.append(header, vector);
    panel.appendChild(box);
    prefixOutputs.push({ kappa, vector });
  }

  const note = document.createElement('p');
  note.className = 'note';
  note.textContent = selected.source
    ? `${selected.source.file} · raw ${selected.source.native_frameskip}-tick frames. ` +
      (selected.source.resampled ? 'Auxiliary fields show the nearest native replay frame to each resampled scene.' :
        'All 161 original replay components are available below.')
    : selected.source_note;
  panel.appendChild(note);
  const sceneTitle = document.createElement('h2');
  sceneTitle.textContent = 'Model input · normalized 51-component scene';
  panel.appendChild(sceneTitle);
  selected.feature_groups.slice(0, 3).forEach((group) => addFeatureGroup(panel, group, 'scene', true));
  if (selected.source) {
    const rawTitle = document.createElement('h2');
    rawTitle.textContent = 'Additional original replay components';
    rawTitle.style.marginTop = '20px';
    panel.appendChild(rawTitle);
    selected.feature_groups.slice(3).forEach((group) => addFeatureGroup(panel, group, 'raw', false));
  }
}
function showFrame(newIndex) {
  if (!skill) return;
  index = Math.max(0, Math.min(newIndex, skill.scenes.length - 1));
  const values = skill.scenes[index];
  showScene(values);
  $('scrub').value = index;
  $('frameLabel').textContent = `Frame ${index + 1} / ${skill.scenes.length}`;
  $('timeLabel').textContent = `${(index * skill.frameskip / 120).toFixed(2)} s` +
    (skill.source ? ` · raw row ${skill.source.native_rows[index]}` : '');
  for (const { output, source, column } of outputs) {
    output.textContent = format(source === 'scene' ? values[column] : skill.source.raw_frames[index][column]);
  }
  for (const [car, output] of prefixOutputs.entries()) {
    output.kappa.textContent = `κ ${skill.prefix_concentrations[index][car].toFixed(3)}`;
    output.vector.textContent = `[${skill.prefix_latents[index][car]
      .map((value) => value.toFixed(4)).join(', ')}]`;
  }
}
function pause() {
  playing = false;
  $('play').textContent = 'Play';
  accumulated = 0;
}
async function selectSkill(id) {
  const request = ++loading;
  pause();
  try {
    const response = await fetch(`/api/skills/${id}`);
    if (!response.ok) throw new Error(`Cannot load skill ${id + 1} (HTTP ${response.status})`);
    const selected = await response.json();
    if (request !== loading) return;
    skill = selected;
    $('skillSelect').value = String(id);
    $('scrub').max = selected.scenes.length - 1;
    inspect(selected);
    showTrails(selected.scenes);
    showFrame(0);
    if (downloadUrl) URL.revokeObjectURL(downloadUrl);
    downloadUrl = URL.createObjectURL(new Blob([JSON.stringify(selected, null, 2)],
      { type: 'application/json' }));
    $('download').href = downloadUrl;
    $('download').download = `expert-skill-${selected.id + 1}.json`;
  } catch (error) {
    if (request !== loading) return;
    const message = document.createElement('p');
    message.className = 'empty';
    message.textContent = error.message;
    $('inspector').replaceChildren(message);
  }
}
function filterSkills() {
  const previous = skill?.id;
  const search = $('skillSearch').value.trim().toLowerCase();
  $('skillSelect').replaceChildren();
  for (const item of allSkills) {
    const text = `skill ${item.id + 1} · sample ${item.chunk + 1} · frames ${item.start}-${item.stop}`;
    if (!text.includes(search)) continue;
    const option = document.createElement('option');
    option.value = item.id;
    option.textContent = `${text} · ${item.duration} steps`;
    $('skillSelect').appendChild(option);
  }
  if (previous !== undefined && [...$('skillSelect').options].some(
    (option) => Number(option.value) === previous)) {
    $('skillSelect').value = String(previous);
  } else if ($('skillSelect').options.length) {
    selectSkill(Number($('skillSelect').value));
  } else {
    pause();
    skill = null;
    loading += 1;
    const message = document.createElement('p');
    message.className = 'empty';
    message.textContent = allSkills.length ? 'No expert skills match this filter.' :
      'This checkpoint has no expert skills.';
    $('inspector').replaceChildren(message);
  }
}

$('skillSearch').addEventListener('input', filterSkills);
$('skillSelect').addEventListener('change', () => selectSkill(Number($('skillSelect').value)));
$('previous').addEventListener('click', () => { pause(); showFrame(index - 1); });
$('next').addEventListener('click', () => { pause(); showFrame(index + 1); });
$('scrub').addEventListener('input', () => { pause(); showFrame(Number($('scrub').value)); });
$('play').addEventListener('click', () => {
  if (!skill) return;
  if (playing) { pause(); return; }
  if (index === skill.scenes.length - 1) showFrame(0);
  playing = true;
  lastTime = performance.now();
  $('play').textContent = 'Pause';
});
addEventListener('keydown', (event) => {
  if (['INPUT', 'SELECT'].includes(document.activeElement.tagName)) return;
  if (event.key === 'ArrowLeft') { pause(); showFrame(index - 1); event.preventDefault(); }
  if (event.key === 'ArrowRight') { pause(); showFrame(index + 1); event.preventDefault(); }
  if (event.key === ' ') { $('play').click(); event.preventDefault(); }
});
addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
async function loadSkills() {
  try {
    const response = await fetch('/api/skills');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const catalog = await response.json();
    allSkills = catalog.skills;
    $('checkpoint').textContent = `${catalog.checkpoint} · ${catalog.step.toLocaleString()} actor-steps · ${allSkills.length} sampled expert skills`;
    filterSkills();
  } catch (error) {
    $('checkpoint').textContent = `Could not load expert skills: ${error.message}`;
  }
}
function render(now) {
  if (playing && skill) {
    accumulated += (now - lastTime) * Number($('speed').value);
    const frameTime = skill.frameskip / 120 * 1000;
    if (accumulated >= frameTime) {
      const count = Math.floor(accumulated / frameTime);
      accumulated %= frameTime;
      showFrame(index + count);
      if (index === skill.scenes.length - 1) pause();
    }
  }
  lastTime = now;
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(render);
}
loadSkills();
requestAnimationFrame(render);
