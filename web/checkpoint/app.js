import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

const view = document.getElementById('view');
const checkpoint = document.getElementById('checkpoint');
const demo = document.getElementById('demo');
const reward = document.getElementById('reward');
const agentBoost = document.getElementById('agent-boost');
const expertBoost = document.getElementById('expert-boost');
const demoSearch = document.getElementById('demo-search');
const demoQuery = document.getElementById('demo-query');
const connection = document.getElementById('connection');
const speed = document.getElementById('speed');
const speedValue = document.getElementById('speed-value');
let speedUpdate;

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
  const material = new THREE.LineBasicMaterial({ color, transparent: true, opacity });
  scene.add(new THREE.Line(geometry, material));
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

function makeCar(color, opacity = 1) {
  const group = new THREE.Group();
  const translucent = opacity < 1;
  const material = new THREE.MeshStandardMaterial({
    color,
    transparent: translucent,
    opacity,
    depthWrite: !translucent,
    polygonOffset: translucent,
    polygonOffsetFactor: -2,
    polygonOffsetUnits: -2,
    roughness: 0.36,
    metalness: 0.08,
  });
  const body = new THREE.Mesh(new THREE.BoxGeometry(120, 87, 39), material);
  body.castShadow = opacity === 1;
  body.receiveShadow = true;
  group.add(body);

  const nose = new THREE.Mesh(
    new THREE.ConeGeometry(17, 38, 3),
    new THREE.MeshStandardMaterial({
      color: 0xffffff,
      transparent: translucent,
      opacity,
      depthWrite: !translucent,
      polygonOffset: translucent,
      polygonOffsetFactor: -2,
      polygonOffsetUnits: -2,
    }),
  );
  nose.rotation.z = -Math.PI / 2;
  nose.position.x = 75;
  group.add(nose);
  group.renderOrder = translucent ? 2 : 1;
  group.userData.bodyMaterial = material;
  scene.add(group);
  return group;
}

const car = makeCar(0x145bd7);
const ghost = makeCar(0x009b82, 0.68);
const ball = new THREE.Mesh(
  new THREE.SphereGeometry(91.25, 28, 20),
  new THREE.MeshStandardMaterial({ color: 0xf8fbfc, roughness: 0.3, metalness: 0.04 }),
);
ball.castShadow = true;
scene.add(ball);
const ghostBall = new THREE.Mesh(
  new THREE.SphereGeometry(94, 24, 16),
  new THREE.MeshStandardMaterial({ color: 0x00b992, transparent: true, opacity: 0.42, wireframe: true }),
);
scene.add(ghostBall);

const forward = new THREE.Vector3();
const right = new THREE.Vector3();
const up = new THREE.Vector3();
const basis = new THREE.Matrix4();

function setCar(mesh, state) {
  mesh.visible = Boolean(state) && !state.demoed;
  if (!state) return;
  mesh.position.fromArray(state.pos);
  forward.fromArray(state.fwd);
  right.fromArray(state.rgt);
  up.fromArray(state.up);
  basis.makeBasis(forward, right, up);
  mesh.quaternion.setFromRotationMatrix(basis);
}

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
  setCar(car, frame.cars[0]);
  setCar(ghost, frame.expert.cars[0]);
  agentBoost.textContent = frame.cars[0].boost.toFixed(0);
  expertBoost.textContent = frame.expert.cars[0].boost.toFixed(0);
  ball.position.fromArray(frame.ball.pos);
  ghostBall.position.fromArray(frame.expert.ball.pos);
  checkpoint.textContent = frame.checkpoint;
  checkpoint.title = frame.checkpoint;
  if (demo.textContent !== frame.demo) {
    demo.textContent = frame.demo;
    demo.title = frame.demo;
  }
  reward.textContent = frame.reward.map((value) => value.toFixed(3)).join(' / ');
};
source.onerror = () => {
  connection.textContent = 'Reconnecting';
  connection.classList.remove('live');
};

document.getElementById('reset').addEventListener('click', () => {
  fetch('/reset', { method: 'POST' });
});
document.getElementById('previous').addEventListener('click', () => {
  fetch('/previous', { method: 'POST' });
});
document.getElementById('next').addEventListener('click', () => {
  fetch('/next', { method: 'POST' });
});
demoSearch.addEventListener('submit', (event) => {
  event.preventDefault();
  fetch('/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: demoQuery.value }),
  });
});
speed.addEventListener('input', () => {
  speedValue.value = `${Number(speed.value).toFixed(1).replace('.0', '')}x`;
  clearTimeout(speedUpdate);
  speedUpdate = setTimeout(() => {
    fetch('/speed', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ speed: Number(speed.value) }),
    });
  }, 50);
});
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
