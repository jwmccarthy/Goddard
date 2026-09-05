import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

const view = document.getElementById('view');
const connection = document.getElementById('connection');
const blueLabel = document.getElementById('blue');
const orangeLabel = document.getElementById('orange');
const goals = document.getElementById('goals');
const roundLabel = document.getElementById('round');
const blueCheckpoint = document.getElementById('blueCheckpoint');
const orangeCheckpoint = document.getElementById('orangeCheckpoint');
const blueBoost = document.getElementById('blue-boost');
const orangeBoost = document.getElementById('orange-boost');
const blueBoostFill = document.getElementById('blue-boost-fill');
const orangeBoostFill = document.getElementById('orange-boost-fill');

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.95;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
view.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xbfd1d6);
scene.fog = new THREE.Fog(0xbfd1d6, 11500, 22000);
scene.add(new THREE.HemisphereLight(0xf5fbfc, 0x577773, 1.9));
const sun = new THREE.DirectionalLight(0xfff4da, 3.2);
sun.position.set(-3500, -4500, 9000);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.left = -7000;
sun.shadow.camera.right = 7000;
sun.shadow.camera.top = 7000;
sun.shadow.camera.bottom = -7000;
scene.add(sun);

const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 10, 30000);
camera.up.set(0, 0, 1);
camera.position.set(5200, -6900, 4300);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 500);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.maxDistance = 18000;
controls.minDistance = 1200;

const field = new THREE.Mesh(
  new THREE.PlaneGeometry(8192, 10240),
  new THREE.MeshStandardMaterial({ color: 0x86ad9d, roughness: 0.92 }),
);
field.position.z = -4;
field.receiveShadow = true;
field.renderOrder = -2;
scene.add(field);

function addLine(points, color = 0xedf5f1, opacity = 0.62) {
  const geometry = new THREE.BufferGeometry().setFromPoints(
    points.map(([x, y]) => new THREE.Vector3(x, y, 8)),
  );
  scene.add(new THREE.Line(
    geometry,
    new THREE.LineBasicMaterial({ color, transparent: true, opacity }),
  ));
}
addLine([[-4096, 0], [4096, 0]]);
addLine([[-4096, -5120], [4096, -5120], [4096, 5120], [-4096, 5120], [-4096, -5120]], 0xffffff, 0.55);
const centerCircle = [];
for (let index = 0; index <= 64; index += 1) {
  const angle = index / 64 * Math.PI * 2;
  centerCircle.push([Math.cos(angle) * 920, Math.sin(angle) * 920]);
}
addLine(centerCircle);
const grid = new THREE.GridHelper(10240, 20, 0x4a8a7d, 0x76aa9c);
grid.rotation.x = Math.PI / 2;
grid.position.z = 3;
grid.material.transparent = true;
grid.material.opacity = 0.18;
scene.add(grid);

new OBJLoader().load('/arena.obj', (arena) => {
  arena.traverse((child) => {
    if (!child.isMesh) return;
    child.material = new THREE.MeshStandardMaterial({
      color: 0x7897a0,
      transparent: true,
      opacity: 0.2,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: 2,
      polygonOffsetUnits: 2,
      roughness: 0.65,
      side: THREE.DoubleSide,
    });
    child.renderOrder = -1;
  });
  arena.rotation.z = Math.PI / 2;
  scene.add(arena);
});

function makeCar(color) {
  const group = new THREE.Group();
  const material = new THREE.MeshStandardMaterial({
    color,
    emissive: color,
    emissiveIntensity: 0,
    roughness: 0.36,
    metalness: 0.08,
  });
  const body = new THREE.Mesh(new THREE.BoxGeometry(120, 87, 39), material);
  body.castShadow = true;
  body.receiveShadow = true;
  group.add(body);
  const nose = new THREE.Mesh(
    new THREE.ConeGeometry(17, 38, 3),
    new THREE.MeshStandardMaterial({ color: 0xffffff }),
  );
  nose.rotation.z = -Math.PI / 2;
  nose.position.x = 75;
  group.add(nose);
  scene.add(group);
  return { group, material };
}

const cars = [makeCar(0x145bd7), makeCar(0xe65b35)];
const ball = new THREE.Mesh(
  new THREE.SphereGeometry(91.25, 28, 20),
  new THREE.MeshStandardMaterial({ color: 0xf8fbfc, roughness: 0.3, metalness: 0.04 }),
);
ball.castShadow = true;
scene.add(ball);
const basis = new THREE.Matrix4();
const forward = new THREE.Vector3();
const right = new THREE.Vector3();
const up = new THREE.Vector3();

function setCar(index, state) {
  const rig = cars[index];
  rig.group.visible = !state.demoed;
  rig.group.position.fromArray(state.pos);
  forward.fromArray(state.fwd);
  right.fromArray(state.rgt);
  up.fromArray(state.up);
  basis.makeBasis(forward, right, up);
  rig.group.quaternion.setFromRotationMatrix(basis);
  rig.material.emissiveIntensity = state.boosting ? 1.2 : 0;
}

let initialSelection = false;
const source = new EventSource('/api/stream');
source.onopen = () => {
  connection.textContent = 'Live';
  connection.classList.add('live');
};
source.onmessage = ({ data }) => {
  const frame = JSON.parse(data);
  if (frame.error) {
    connection.textContent = frame.error;
    connection.classList.remove('live');
    return;
  }
  setCar(0, frame.cars[0]);
  setCar(1, frame.cars[1]);
  ball.position.fromArray(frame.ball.pos);
  blueLabel.textContent = frame.blue.checkpoint;
  orangeLabel.textContent = frame.orange.checkpoint;
  goals.textContent = `${frame.blue.score} - ${frame.orange.score}`;
  roundLabel.textContent = `round ${frame.round} | tick ${frame.tick}`;
  blueBoost.textContent = frame.cars[0].boost.toFixed(0);
  orangeBoost.textContent = frame.cars[1].boost.toFixed(0);
  blueBoostFill.style.width = `${frame.cars[0].boost}%`;
  orangeBoostFill.style.width = `${frame.cars[1].boost}%`;
  if (!initialSelection) {
    blueCheckpoint.value = frame.blue.path;
    orangeCheckpoint.value = frame.orange.path;
    initialSelection = true;
  }
};
source.onerror = () => {
  connection.textContent = 'Reconnecting';
  connection.classList.remove('live');
};

async function refreshCheckpoints() {
  const previous = [blueCheckpoint.value, orangeCheckpoint.value];
  const checkpoints = await fetch('/api/checkpoints').then((response) => response.json());
  for (const select of [blueCheckpoint, orangeCheckpoint]) select.replaceChildren();
  for (const checkpoint of checkpoints) {
    for (const select of [blueCheckpoint, orangeCheckpoint]) {
      const option = document.createElement('option');
      option.value = checkpoint.path;
      option.textContent = `${checkpoint.label} (${checkpoint.step.toLocaleString()})`;
      select.appendChild(option);
    }
  }
  if (checkpoints.some((item) => item.path === previous[0])) blueCheckpoint.value = previous[0];
  if (checkpoints.some((item) => item.path === previous[1])) orangeCheckpoint.value = previous[1];
}

document.getElementById('applyMatch').addEventListener('click', () => fetch('/api/match', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ blue: blueCheckpoint.value, orange: orangeCheckpoint.value }),
}));
document.getElementById('resetMatch').addEventListener('click', () => {
  fetch('/api/reset', { method: 'POST' });
});
refreshCheckpoints();
setInterval(refreshCheckpoints, 5000);
addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
function render() {
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(render);
}
render();
