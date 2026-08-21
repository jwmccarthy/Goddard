import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

const view = document.getElementById('view');
const status = document.getElementById('status');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(devicePixelRatio);
view.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x070a10);
scene.add(new THREE.HemisphereLight(0xbcd9ff, 0x151922, 2));

const camera = new THREE.PerspectiveCamera(58, innerWidth / innerHeight, 10, 30000);
camera.up.set(0, 0, 1);
camera.position.set(4300, -5600, 3200);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 450);
controls.enableDamping = true;

new OBJLoader().load('/arena.obj', (arena) => {
  arena.traverse((child) => {
    if (!child.isMesh) return;
    child.material = new THREE.MeshStandardMaterial({ color: 0x263143, transparent: true, opacity: 0.45, side: THREE.DoubleSide });
  });
  arena.rotation.z = Math.PI / 2;
  scene.add(arena);
});

function makeCar(color, opacity = 1) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(120, 87, 39), new THREE.MeshStandardMaterial({ color, transparent: opacity < 1, opacity }));
  scene.add(mesh);
  return mesh;
}

const cars = [makeCar(0x168cff), makeCar(0xff7028)];
const ball = new THREE.Mesh(new THREE.SphereGeometry(91.25, 24, 16), new THREE.MeshStandardMaterial({ color: 0xe9edf2 }));
scene.add(ball);
const ghosts = [makeCar(0x74c6ff, 0.3), makeCar(0xffbc90, 0.3)];
const ghostBall = new THREE.Mesh(new THREE.SphereGeometry(91.25, 24, 16), new THREE.MeshStandardMaterial({ color: 0xc8ffea, transparent: true, opacity: 0.3 }));
scene.add(ghostBall);

const forward = new THREE.Vector3();
const right = new THREE.Vector3();
const up = new THREE.Vector3();
const basis = new THREE.Matrix4();

function setCar(mesh, state) {
  mesh.visible = !state.demoed;
  mesh.position.fromArray(state.pos);
  forward.fromArray(state.fwd);
  right.fromArray(state.rgt);
  up.fromArray(state.up);
  basis.makeBasis(forward, right, up);
  mesh.quaternion.setFromRotationMatrix(basis);
}

const source = new EventSource('/api/stream');
source.onmessage = ({ data }) => {
  const frame = JSON.parse(data);
  if (frame.error) { status.textContent = frame.error; return; }
  setCar(cars[0], frame.cars[0]);
  setCar(cars[1], frame.cars[1]);
  setCar(ghosts[0], frame.expert.cars[0]);
  setCar(ghosts[1], frame.expert.cars[1]);
  ball.position.fromArray(frame.ball.pos);
  ghostBall.position.fromArray(frame.expert.ball.pos);
  status.textContent = `${frame.checkpoint} | reward ${frame.reward.map((value) => value.toFixed(3)).join(' / ')}`;
};
source.onerror = () => { status.textContent = 'reconnecting'; };

document.getElementById('reset').addEventListener('click', () => fetch('/reset', { method: 'POST' }));
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
