import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

const holder = document.getElementById('view');
const status = document.getElementById('status');
const blueLabel = document.getElementById('blue');
const orangeLabel = document.getElementById('orange');
const goals = document.getElementById('goals');
const roundLabel = document.getElementById('round');
const checkpointDirectory = document.getElementById('checkpointDirectory');
const blueCheckpoint = document.getElementById('blueCheckpoint');
const orangeCheckpoint = document.getElementById('orangeCheckpoint');
const sampleActions = document.getElementById('sampleActions');
const replayResets = document.getElementById('replayResets');
const applyMatch = document.getElementById('applyMatch');

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
holder.appendChild(renderer.domElement);

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
controls.dampingFactor = 0.08;
controls.minDistance = 500;
controls.maxDistance = 15000;

const heldKeys = new Set();
const moveForward = new THREE.Vector3();
const moveRight = new THREE.Vector3();
const moveDelta = new THREE.Vector3();
window.addEventListener('keydown', (event) => {
  if (event.code === 'KeyR' && !event.repeat) {
    event.preventDefault();
    fetch('/api/reset', { method: 'POST' });
    return;
  }
  if (['KeyW', 'KeyA', 'KeyS', 'KeyD', 'Space', 'ControlLeft', 'ControlRight'].includes(event.code)) {
    event.preventDefault();
    heldKeys.add(event.code);
  }
});
window.addEventListener('keyup', (event) => heldKeys.delete(event.code));
window.addEventListener('blur', () => heldKeys.clear());

function moveCamera(dt) {
  camera.getWorldDirection(moveForward);
  moveForward.z = 0;
  if (moveForward.lengthSq() < 1e-6) return;
  moveForward.normalize();
  moveRight.crossVectors(moveForward, camera.up).normalize();
  moveDelta.set(0, 0, 0);
  if (heldKeys.has('KeyW')) moveDelta.add(moveForward);
  if (heldKeys.has('KeyS')) moveDelta.sub(moveForward);
  if (heldKeys.has('KeyD')) moveDelta.add(moveRight);
  if (heldKeys.has('KeyA')) moveDelta.sub(moveRight);
  if (heldKeys.has('Space')) moveDelta.z += 1;
  if (heldKeys.has('ControlLeft') || heldKeys.has('ControlRight')) moveDelta.z -= 1;
  if (moveDelta.lengthSq() === 0) return;
  moveDelta.normalize().multiplyScalar(2500 * dt);
  camera.position.add(moveDelta);
  controls.target.add(moveDelta);
}

new OBJLoader().load('/arena.obj', (arena) => {
  arena.traverse((child) => {
    if (!child.isMesh) return;
    child.geometry.computeVertexNormals();
    child.material = new THREE.MeshStandardMaterial({
      color: 0x263143,
      roughness: 1,
      transparent: true,
      opacity: 0.48,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const edgeGeometry = new THREE.EdgesGeometry(child.geometry, 24);
    const positions = edgeGeometry.getAttribute('position');
    const colors = new Float32Array(positions.count * 3);
    const blueLine = new THREE.Color(0x477da8);
    const orangeLine = new THREE.Color(0xa86d47);
    const neutralLine = new THREE.Color(0x52647d);
    for (let index = 0; index < positions.count; index += 2) {
      // The arena's local X axis becomes field Y after its rotation below.
      const midpoint = (positions.getX(index) + positions.getX(index + 1)) / 2;
      const color = midpoint < -50 ? blueLine : midpoint > 50 ? orangeLine : neutralLine;
      for (const vertex of [index, index + 1]) color.toArray(colors, vertex * 3);
    }
    edgeGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    const edges = new THREE.LineSegments(
      edgeGeometry,
      new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.32 })
    );
    child.add(edges);
  });
  arena.rotation.z = Math.PI / 2;
  scene.add(arena);
});

function makeCar(color) {
  const group = new THREE.Group();
  const bodyGeometry = new THREE.BoxGeometry(120.507, 86.699, 38.659);
  const bodyMaterial = new THREE.MeshStandardMaterial({
    color,
    emissive: color,
    emissiveIntensity: 0,
    roughness: 0.45,
  });
  const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
  group.add(body);
  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(bodyGeometry),
    new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.7 })
  );
  group.add(edges);
  const nose = new THREE.Mesh(
    new THREE.ConeGeometry(10, 34, 8),
    new THREE.MeshStandardMaterial({ color: 0xffffff })
  );
  nose.rotation.z = -Math.PI / 2;
  nose.position.x = 78;
  group.add(nose);
  const boostLight = new THREE.PointLight(color, 0, 500, 2);
  boostLight.position.set(-80, 0, 10);
  group.add(boostLight);
  scene.add(group);
  return { group, body, boostLight };
}

const cars = [makeCar(0x168cff), makeCar(0xff7028)];
const ball = new THREE.Mesh(
  new THREE.SphereGeometry(91.25, 32, 20),
  new THREE.MeshStandardMaterial({ color: 0xe9edf2, roughness: 0.7 })
);
scene.add(ball);
const ballShadow = new THREE.Mesh(
  new THREE.CircleGeometry(91.25, 32),
  new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.35, depthWrite: false })
);
ballShadow.position.z = 3;
scene.add(ballShadow);

const basis = new THREE.Matrix4();
const fwd = new THREE.Vector3();
const right = new THREE.Vector3();
const up = new THREE.Vector3();
function setCar(index, state) {
  const rig = cars[index];
  rig.group.visible = !state.demoed;
  rig.group.position.fromArray(state.pos);
  fwd.fromArray(state.fwd);
  right.fromArray(state.rgt);
  up.fromArray(state.up);
  basis.makeBasis(fwd, right, up);
  rig.group.quaternion.setFromRotationMatrix(basis);
  rig.body.material.emissiveIntensity = state.boosting ? 1.4 : 0;
  rig.boostLight.intensity = state.boosting ? 180 : 0;
}

const source = new EventSource('/api/stream');
let initializedSelection = false;
source.onmessage = (event) => {
  const frame = JSON.parse(event.data);
  if (frame.error) {
    status.style.display = 'grid';
    status.textContent = frame.error;
    return;
  }
  status.style.display = 'none';
  setCar(0, frame.cars[0]);
  setCar(1, frame.cars[1]);
  ball.position.fromArray(frame.ball.pos);
  ballShadow.position.x = frame.ball.pos[0];
  ballShadow.position.y = frame.ball.pos[1];
  blueLabel.textContent = `BLUE  ${frame.blue.checkpoint}`;
  orangeLabel.textContent = `${frame.orange.checkpoint}  ORANGE`;
  goals.textContent = `${frame.blue.score} - ${frame.orange.score}`;
  roundLabel.textContent = `round ${frame.round}  |  tick ${frame.tick}`;
  if (!initializedSelection) {
    const blueDirectory = directoryOf(frame.blue.path);
    const orangeDirectory = directoryOf(frame.orange.path);
    if (blueDirectory === orangeDirectory && checkpointDirectory.value !== blueDirectory) {
      checkpointDirectory.value = blueDirectory;
      populateCheckpointMenus();
    }
    const blueOption = [...blueCheckpoint.options].find((option) => option.value === frame.blue.path);
    const orangeOption = [...orangeCheckpoint.options].find((option) => option.value === frame.orange.path);
    if (blueOption && orangeOption) {
      blueCheckpoint.value = blueOption.value;
      orangeCheckpoint.value = orangeOption.value;
      sampleActions.checked = frame.sample_actions;
      replayResets.checked = frame.replay_resets;
      initializedSelection = true;
    }
  }
};
source.onerror = () => {
  status.style.display = 'grid';
  status.textContent = 'reconnecting';
};

let checkpointSignature = '';
let checkpointInventory = [];

function directoryOf(path) {
  const parts = path.split('/');
  parts.pop();
  return parts.join('/') || '.';
}

function populateCheckpointMenus() {
  const directory = checkpointDirectory.value;
  const selections = [blueCheckpoint.value, orangeCheckpoint.value];
  const checkpoints = checkpointInventory.filter((checkpoint) => directoryOf(checkpoint.path) === directory);
  for (const select of [blueCheckpoint, orangeCheckpoint]) select.replaceChildren();
  for (const checkpoint of checkpoints) {
    for (const select of [blueCheckpoint, orangeCheckpoint]) {
      const option = document.createElement('option');
      option.value = checkpoint.path;
      option.textContent = checkpoint.path.split('/').at(-1);
      select.appendChild(option);
    }
  }
  if (checkpoints.some((checkpoint) => checkpoint.path === selections[0])) blueCheckpoint.value = selections[0];
  if (checkpoints.some((checkpoint) => checkpoint.path === selections[1])) orangeCheckpoint.value = selections[1];
}

async function refreshCheckpoints() {
  const checkpoints = await fetch('/api/checkpoints').then((response) => response.json());
  const signature = checkpoints.map((checkpoint) => `${checkpoint.path}:${checkpoint.modified}`).join('|');
  if (signature === checkpointSignature) return;
  checkpointSignature = signature;
  checkpointInventory = checkpoints;

  const selectedDirectory = checkpointDirectory.value;
  const directories = [...new Set(checkpoints.map((checkpoint) => directoryOf(checkpoint.path)))];
  checkpointDirectory.replaceChildren();
  for (const directory of directories) {
    const option = document.createElement('option');
    option.value = directory;
    option.textContent = directory;
    checkpointDirectory.appendChild(option);
  }
  if (directories.includes(selectedDirectory)) checkpointDirectory.value = selectedDirectory;
  populateCheckpointMenus();
}
refreshCheckpoints();
setInterval(refreshCheckpoints, 5000);
checkpointDirectory.addEventListener('change', () => {
  initializedSelection = true;
  populateCheckpointMenus();
});

applyMatch.addEventListener('click', async () => {
  applyMatch.disabled = true;
  try {
    const response = await fetch('/api/match', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        blue: blueCheckpoint.value,
        orange: orangeCheckpoint.value,
        sample_actions: sampleActions.checked,
        replay_resets: replayResets.checked,
      }),
    });
    if (!response.ok) throw new Error(await response.text());
    initializedSelection = true;
  } catch (error) {
    status.style.display = 'grid';
    status.textContent = error.message;
  } finally {
    applyMatch.disabled = false;
  }
});

window.addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

let previousFrame = performance.now();
function render(now) {
  const dt = Math.min((now - previousFrame) / 1000, 0.05);
  previousFrame = now;
  moveCamera(dt);
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(render);
}
requestAnimationFrame(render);
