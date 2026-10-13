import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

const view = document.getElementById('view');
const connection = document.getElementById('connection');
const blueLabel = document.getElementById('blue');
const orangeLabel = document.getElementById('orange');
const goals = document.getElementById('goals');
const objectivesLabel = document.getElementById('objectives');
const blueCheckpoint = document.getElementById('blueCheckpoint');
const orangeCheckpoint = document.getElementById('orangeCheckpoint');
const cameraMode = document.getElementById('cameraMode');
const resetMatch = document.getElementById('resetMatch');
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

for (const [y, color] of [[-2560, 0x7faaa8], [2560, 0xaaa184]]) {
  const fieldHalf = new THREE.Mesh(
    new THREE.PlaneGeometry(8192, 5120),
    new THREE.MeshStandardMaterial({ color, roughness: 0.92 }),
  );
  fieldHalf.position.set(0, y, -4);
  fieldHalf.receiveShadow = true;
  fieldHalf.renderOrder = -2;
  scene.add(fieldHalf);
}

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
  return {
    group, material, initialized: false,
    startPosition: new THREE.Vector3(), targetPosition: new THREE.Vector3(),
    startRotation: new THREE.Quaternion(), targetRotation: new THREE.Quaternion(),
  };
}

const cars = [];
const carColors = [
  [0x145bd7, 0x4895f5, 0x0b3479],
  [0xe65b35, 0xffa365, 0xa54125],
];
const ball = new THREE.Mesh(
  new THREE.SphereGeometry(91.25, 28, 20),
  new THREE.MeshStandardMaterial({ color: 0xf8fbfc, roughness: 0.3, metalness: 0.04 }),
);
ball.castShadow = true;
scene.add(ball);
const ballStart = new THREE.Vector3();
const ballTarget = new THREE.Vector3();
let ballInitialized = false;
const padShapes = {
  large: {
    base: new THREE.CylinderGeometry(116, 116, 6, 32),
    ring: new THREE.RingGeometry(76, 104, 32),
    orb: new THREE.SphereGeometry(46, 16, 12),
  },
  small: {
    base: new THREE.CylinderGeometry(58, 58, 4, 24),
    ring: new THREE.RingGeometry(32, 51, 24),
    orb: new THREE.OctahedronGeometry(19),
  },
};
const padMaterials = {
  ready: new THREE.MeshStandardMaterial({
    color: 0xffb452, emissive: 0xe67c1d, emissiveIntensity: 0.8, roughness: 0.36,
  }),
  recharging: new THREE.MeshStandardMaterial({ color: 0x597174, roughness: 0.85 }),
  readyRing: new THREE.MeshBasicMaterial({ color: 0xffd080 }),
  rechargingRing: new THREE.MeshBasicMaterial({ color: 0x779197 }),
  orb: new THREE.MeshBasicMaterial({ color: 0xffd077 }),
};
const boostPads = [];

function makeBoostPad(pad) {
  const group = new THREE.Group();
  const shape = pad.large ? padShapes.large : padShapes.small;
  const base = new THREE.Mesh(shape.base, padMaterials.ready);
  base.rotation.x = Math.PI / 2;
  base.position.z = 6;
  group.add(base);
  const ring = new THREE.Mesh(shape.ring, padMaterials.readyRing);
  ring.position.z = 10;
  group.add(ring);
  const orb = new THREE.Mesh(shape.orb, padMaterials.orb);
  orb.position.z = pad.large ? pad.pos[2] : 27;
  group.add(orb);
  scene.add(group);
  return { group, base, ring, orb };
}

function setBoostPad(index, pad) {
  if (!boostPads[index]) boostPads[index] = makeBoostPad(pad);
  const rig = boostPads[index];
  rig.group.visible = true;
  rig.group.position.set(pad.pos[0], pad.pos[1], 0);
  rig.base.material = pad.active ? padMaterials.ready : padMaterials.recharging;
  rig.ring.material = pad.active ? padMaterials.readyRing : padMaterials.rechargingRing;
  rig.orb.visible = pad.active;
}

const basis = new THREE.Matrix4();
const forward = new THREE.Vector3();
const right = new THREE.Vector3();
const up = new THREE.Vector3();

function setCar(index, state, snap) {
  const rig = cars[index];
  const wasVisible = rig.group.visible;
  rig.group.visible = !state.demoed;
  rig.startPosition.copy(rig.group.position);
  rig.targetPosition.fromArray(state.pos);
  forward.fromArray(state.fwd);
  right.fromArray(state.rgt);
  up.fromArray(state.up);
  basis.makeBasis(forward, right, up);
  rig.startRotation.copy(rig.group.quaternion);
  rig.targetRotation.setFromRotationMatrix(basis);
  if (snap || !rig.initialized || !wasVisible
      || rig.group.position.distanceToSquared(rig.targetPosition) > 1000 ** 2) {
    rig.group.position.copy(rig.targetPosition);
    rig.group.quaternion.copy(rig.targetRotation);
    rig.startPosition.copy(rig.targetPosition);
    rig.startRotation.copy(rig.targetRotation);
  }
  rig.initialized = true;
  rig.material.emissiveIntensity = state.boosting ? 1.2 : 0;
}

let poseStartedAt = performance.now();
let poseDuration = 1000 / 30;
function updatePoses(time) {
  const fraction = Math.max(0, Math.min(1, (time - poseStartedAt) / poseDuration));
  for (const rig of cars) {
    rig.group.position.lerpVectors(rig.startPosition, rig.targetPosition, fraction);
    rig.group.quaternion.slerpQuaternions(rig.startRotation, rig.targetRotation, fraction);
  }
  ball.position.lerpVectors(ballStart, ballTarget, fraction);
}

let cameraFrame = null;
let cameraCars = '';
let activeCameraMode = 'free';
let initializeFollow = false;
const orbitPosition = camera.position.clone();
const orbitTarget = controls.target.clone();
const followTarget = new THREE.Vector3();
const followDelta = new THREE.Vector3();
const followHeading = new THREE.Vector3();

function changeCameraMode() {
  const mode = cameraMode.value;
  if (mode === 'free') {
    camera.position.copy(orbitPosition);
    controls.target.copy(orbitTarget);
    controls.enablePan = true;
    controls.minDistance = 1200;
    controls.update();
  } else {
    if (activeCameraMode === 'free') {
      orbitPosition.copy(camera.position);
      orbitTarget.copy(controls.target);
    }
    controls.enablePan = false;
    controls.minDistance = 250;
    initializeFollow = true;
  }
  activeCameraMode = mode;
}
cameraMode.addEventListener('change', changeCameraMode);

function updateCameraChoices(players) {
  const signature = players.map((car) => `${car.team}:${car.player}`).join(',');
  if (signature === cameraCars) return;
  cameraCars = signature;
  for (const option of cameraMode.querySelectorAll('option[data-car]')) option.remove();
  for (const car of players) {
    const option = document.createElement('option');
    option.value = `car:${car.team}:${car.player}`;
    option.textContent = `Follow ${car.team ? 'Orange' : 'Blue'} ${car.player}`;
    option.dataset.car = '';
    cameraMode.appendChild(option);
  }
  if (![...cameraMode.options].some((option) => option.value === activeCameraMode)) {
    cameraMode.value = 'free';
    changeCameraMode();
  }
}

function updateFollowCamera() {
  if (!cameraFrame) return;
  let car, rig;
  if (cameraMode.value.startsWith('car:')) {
    const [, team, player] = cameraMode.value.split(':');
    const index = cameraFrame.cars.findIndex(
      (item) => item.team === Number(team) && item.player === Number(player),
    );
    if (index >= 0 && !cameraFrame.cars[index].demoed) {
      car = cameraFrame.cars[index];
      rig = cars[index];
    }
  }
  // Follow the ball while the selected car is demolished or unavailable.
  followTarget.copy(rig ? rig.group.position : ball.position);
  if (rig) followTarget.z += 80;
  if (initializeFollow) {
    controls.target.copy(followTarget);
    if (car) {
      const horizontal = Math.hypot(car.fwd[0], car.fwd[1]);
      followHeading.set(
        horizontal > 0.2 ? car.fwd[0] / horizontal : 0,
        horizontal > 0.2 ? car.fwd[1] / horizontal : (car.team ? -1 : 1), 0,
      );
      camera.position.copy(followTarget).addScaledVector(followHeading, -750);
      camera.position.z += 250;
    } else {
      camera.position.set(followTarget.x, followTarget.y - 1300, followTarget.z + 900);
    }
    initializeFollow = false;
  } else {
    // Move the camera with the interpolated subject, preserving the angle and
    // zoom that OrbitControls (including a user's drag) chose around it.
    followDelta.subVectors(followTarget, controls.target);
    camera.position.add(followDelta);
    controls.target.copy(followTarget);
  }
}

let initialSelection = false;
const source = new EventSource('/api/stream');
source.onopen = () => {
  connection.hidden = true;
};
source.onmessage = ({ data }) => {
  const frame = JSON.parse(data);
  if (frame.error) {
    connection.textContent = frame.error;
    connection.hidden = false;
    resetMatch.disabled = true;
    return;
  }
  connection.hidden = true;
  resetMatch.disabled = false;
  const receivedAt = performance.now();
  updatePoses(receivedAt);
  const snap = !cameraFrame || frame.round !== cameraFrame.round || frame.tick < cameraFrame.tick
    || frame.cars.length !== cameraFrame.cars.length
    || frame.blue.path !== cameraFrame.blue.path || frame.orange.path !== cameraFrame.orange.path;
  if (cameraFrame && !snap && frame.tick > cameraFrame.tick) {
    poseDuration = Math.max(12, Math.min(100, (frame.tick - cameraFrame.tick) * 1000 / 120));
  }
  poseStartedAt = receivedAt;
  cameraFrame = frame;
  updateCameraChoices(frame.cars);
  for (const [index, car] of frame.cars.entries()) {
    if (!cars[index]) cars[index] = makeCar(carColors[car.team][(car.player - 1) % 3]);
    setCar(index, car, snap);
  }
  for (const car of cars.slice(frame.cars.length)) car.group.visible = false;
  for (const [index, pad] of frame.boostPads.entries()) setBoostPad(index, pad);
  for (const pad of boostPads.slice(frame.boostPads.length)) pad.group.visible = false;
  const orangeIndex = frame.cars.length / 2;
  ballStart.copy(ball.position);
  ballTarget.fromArray(frame.ball.pos);
  if (snap || !ballInitialized || ball.position.distanceToSquared(ballTarget) > 900 ** 2) {
    ball.position.copy(ballTarget);
    ballStart.copy(ballTarget);
  }
  ballInitialized = true;
  blueLabel.textContent = frame.blue.checkpoint;
  orangeLabel.textContent = frame.orange.checkpoint;
  goals.textContent = `${frame.blue.score} - ${frame.orange.score}`;
  const objectives = [frame.blue.objective, frame.orange.objective]
    .map((objective, index) => objective ? `${index ? 'orange' : 'blue'}: ${objective}` : null)
    .filter(Boolean);
  objectivesLabel.textContent = objectives.join(' | ');
  objectivesLabel.hidden = objectives.length === 0;
  blueBoost.textContent = frame.cars[0].boost.toFixed(0);
  orangeBoost.textContent = frame.cars[orangeIndex].boost.toFixed(0);
  blueBoostFill.style.width = `${frame.cars[0].boost}%`;
  orangeBoostFill.style.width = `${frame.cars[orangeIndex].boost}%`;
  if (!initialSelection) {
    blueCheckpoint.value = frame.blue.path;
    orangeCheckpoint.value = frame.orange.path;
    initialSelection = true;
  }
};
source.onerror = () => {
  connection.textContent = 'Reconnecting';
  connection.hidden = false;
};

async function refreshCheckpoints() {
  const previous = [blueCheckpoint.value, orangeCheckpoint.value];
  const checkpoints = await fetch('/api/checkpoints').then((response) => response.json());
  for (const select of [blueCheckpoint, orangeCheckpoint]) select.replaceChildren();
  for (const checkpoint of checkpoints) {
    for (const select of [blueCheckpoint, orangeCheckpoint]) {
      const option = document.createElement('option');
      option.value = checkpoint.path;
      option.textContent = `${checkpoint.label} (${checkpoint.kind}, ${checkpoint.step.toLocaleString()})`;
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
resetMatch.addEventListener('click', () => fetch('/api/reset', { method: 'POST' }));
document.getElementById('kickoffMatch').addEventListener('click', () => {
  fetch('/api/kickoff', { method: 'POST' });
});
refreshCheckpoints();
setInterval(refreshCheckpoints, 5000);
addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
function render(time) {
  updatePoses(time);
  if (cameraMode.value !== 'free') updateFollowCamera();
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(render);
}
requestAnimationFrame(render);
