import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

const view = document.getElementById('view');
const status = document.getElementById('status');
const blueLabel = document.getElementById('blue');
const orangeLabel = document.getElementById('orange');
const goals = document.getElementById('goals');
const roundLabel = document.getElementById('round');
const blueCheckpoint = document.getElementById('blueCheckpoint');
const orangeCheckpoint = document.getElementById('orangeCheckpoint');

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
view.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x070a10);
scene.fog = new THREE.Fog(0x070a10, 9000, 18000);
scene.add(new THREE.HemisphereLight(0xbcd9ff, 0x151922, 1.5));
const sun = new THREE.DirectionalLight(0xffffff, 1.5);
sun.position.set(-2500, -3500, 6000);
scene.add(sun);

const camera = new THREE.PerspectiveCamera(58, innerWidth / innerHeight, 10, 30000);
camera.up.set(0, 0, 1);
camera.position.set(4300, -5600, 3200);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 450);
controls.enableDamping = true;
controls.minDistance = 500;
controls.maxDistance = 15000;

new OBJLoader().load('/arena.obj', (arena) => {
  arena.traverse((child) => {
    if (!child.isMesh) return;
    child.material = new THREE.MeshStandardMaterial({ color: 0x263143, roughness: 1, transparent: true, opacity: 0.48, side: THREE.DoubleSide, depthWrite: false });
  });
  arena.rotation.z = Math.PI / 2;
  scene.add(arena);
});

function makeCar(color) {
  const group = new THREE.Group();
  const body = new THREE.Mesh(
    new THREE.BoxGeometry(120.507, 86.699, 38.659),
    new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0, roughness: 0.45 }),
  );
  group.add(body);
  const nose = new THREE.Mesh(new THREE.ConeGeometry(10, 34, 8), new THREE.MeshStandardMaterial({ color: 0xffffff }));
  nose.rotation.z = -Math.PI / 2;
  nose.position.x = 78;
  group.add(nose);
  scene.add(group);
  return { group, body };
}

const cars = [makeCar(0x168cff), makeCar(0xff7028)];
const ball = new THREE.Mesh(new THREE.SphereGeometry(91.25, 32, 20), new THREE.MeshStandardMaterial({ color: 0xe9edf2, roughness: 0.7 }));
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
  rig.body.material.emissiveIntensity = state.boosting ? 1.4 : 0;
}

let initialSelection = false;
const source = new EventSource('/api/stream');
source.onmessage = ({ data }) => {
  const frame = JSON.parse(data);
  if (frame.error) {
    status.style.display = 'grid';
    status.textContent = frame.error;
    return;
  }
  status.style.display = 'none';
  setCar(0, frame.cars[0]);
  setCar(1, frame.cars[1]);
  ball.position.fromArray(frame.ball.pos);
  blueLabel.textContent = `BLUE  ${frame.blue.checkpoint}`;
  orangeLabel.textContent = `${frame.orange.checkpoint}  ORANGE`;
  goals.textContent = `${frame.blue.score} - ${frame.orange.score}`;
  roundLabel.textContent = `round ${frame.round} | tick ${frame.tick}`;
  if (!initialSelection) {
    blueCheckpoint.value = frame.blue.path;
    orangeCheckpoint.value = frame.orange.path;
    initialSelection = true;
  }
};
source.onerror = () => {
  status.style.display = 'grid';
  status.textContent = 'reconnecting';
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

document.getElementById('applyMatch').addEventListener('click', async () => {
  await fetch('/api/match', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ blue: blueCheckpoint.value, orange: orangeCheckpoint.value }),
  });
});
document.getElementById('resetMatch').addEventListener('click', () => fetch('/api/reset', { method: 'POST' }));
refreshCheckpoints();
setInterval(refreshCheckpoints, 5000);

window.addEventListener('resize', () => {
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
