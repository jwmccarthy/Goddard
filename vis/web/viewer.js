import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';
import { OBJLoader } from './vendor/OBJLoader.js';

const viewport = document.querySelector('#viewport');
const boostGauge = document.querySelector('#boost-gauge');
const boostValue = document.querySelector('#boost-value');
const trainingInfo = document.querySelector('#training-info');
const trainingStages = document.querySelector('#training-stages');
const controlsInfo = document.querySelector('#controls-info');

let trainingLayout = '';

function renderTraining(training) {
  trainingInfo.hidden = !training;
  if (!training) return;

  const layout = JSON.stringify([
    training.exercise, training.stage, training.stages, training.failures,
    training.terminal, training.outcome
  ]);

  if (layout !== trainingLayout) {
    trainingLayout = layout;
    trainingStages.replaceChildren();

    training.stages.forEach((stage, index) => {
      const current = index === training.stage - 1;
      const completed = index < training.stage - 1 || (training.terminal && training.outcome === 'success');
      const failed = current && training.terminal && ['failure', 'timeout'].includes(training.outcome);
      const item = document.createElement('li');
      item.className = `training-stage${completed ? ' complete' : current ? ' current' : ''}${failed ? ' failed' : ''}`;
      if (current && !completed) item.setAttribute('aria-current', 'step');

      const content = document.createElement('div');
      content.className = 'training-content';
      const label = document.createElement('div');
      label.className = 'training-label';
      label.textContent = stage.label.replace(' · speed change ≥ 100', '');
      label.title = stage.label;
      content.append(label);

      if (stage.failures.length) {
        item.title = `Avoid: ${stage.failures.join(', ')}`;
      }

      const marker = document.createElement('span');
      marker.className = 'training-marker';
      if (stage.deadline !== null) {
        marker.classList.add('timing-ring');
        marker.style.setProperty('--minimum', `${Math.min(100, 100 * stage.min_ticks / stage.deadline)}%`);
        marker.title = `${(stage.min_ticks / 120).toFixed(2)}–${(stage.deadline / 120).toFixed(2)}s from ${stage.from_exercise_start ? 'reset' : 'stage start'}`;
      }
      if (current && !completed) marker.classList.add('active');
      item.append(content, marker);
      trainingStages.append(item);
    });

  }

  const currentStage = training.stages[training.stage - 1];
  const ring = trainingStages.querySelector('.training-stage.current .timing-ring.active');
  if (ring && currentStage.deadline > 0) {
    ring.style.setProperty('--progress', `${Math.max(0, Math.min(1, training.elapsed / currentStage.deadline)) * 360}deg`);
  }
}

function createCar(color, y, yaw) {
  const car = new THREE.Group();
  const paint = new THREE.MeshStandardMaterial({ color });
  const body = new THREE.Mesh(new THREE.BoxGeometry(120.5, 86.7, 38.7), paint);
  body.position.set(13.9, 0, 20.75);
  car.add(body);

  const tires = new THREE.MeshStandardMaterial({ color: 0x15191e });
  for (const [x, width, radius] of [[51.25, 25.9, 12.5], [-33.75, 29.5, 15]]) {
    for (const side of [-1, 1]) {
      const wheel = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, 12, 16), tires);
      wheel.position.set(x, side * width, radius - 17);
      car.add(wheel);
    }
  }
  car.position.set(0, y, 17);
  car.rotation.z = yaw;
  return car;
}

async function start() {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  viewport.append(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b1220);
  scene.add(new THREE.HemisphereLight(0xe2efff, 0x344154, 3));

  // Rocket League is Z-up: rotate (x, y, z) into browser (x, z, -y).
  const world = new THREE.Group();
  world.rotation.x = -Math.PI / 2;
  scene.add(world);
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(8192, 10240),
    new THREE.MeshStandardMaterial({ color: 0x173c3b })
  );
  floor.position.z = -2;
  world.add(floor);
  const cars = new Map();
  const pads = [];
  const frames = [];
  let carStates = [];
  const ball = new THREE.Mesh(
    new THREE.SphereGeometry(91.25, 24, 16),
    new THREE.MeshStandardMaterial({ color: 0xf0eee4 })
  );
  ball.position.set(0, 0, 93.15);
  world.add(ball);

  const camera = new THREE.PerspectiveCamera(55, 1, 5, 60000);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.minDistance = 120;
  controls.maxDistance = 26000;
  controls.maxPolarAngle = Math.PI / 2 - 0.02;
  controls.zoomToCursor = true;
  let follow = null;

  function setCamera(name) {
    follow = name === 'overview' ? null : name === 'blue' ? 0 : 1;
    if (follow === null) boostGauge.hidden = true;
    const direction = { blue: 1, orange: -1 }[name];
    if (direction) {
      camera.position.set(500, 450, direction * 2500);
      controls.target.set(0, 60, direction * 1600);
    } else {
      camera.position.set(9200, 10000, 11500);
      controls.target.set(0, 0, 0);
    }
    controls.update();
  }
  window.addEventListener('keydown', (event) => {
    const name = { '0': 'overview', '1': 'blue', '2': 'orange' }[event.key];
    if (name) setCamera(name);
    const reset = { r: 'replay', k: 'kickoff', p: 'pack', n: 'next', b: 'previous' }[event.key.toLowerCase()];
    if (reset && !event.repeat) {
      fetch(`/reset/${reset}`, { method: 'POST' }).then((response) => {
        if (!response.ok) console.error(`Reset failed: ${response.status}`);
      }).catch((error) => console.error('Reset failed:', error));
    }
  });
  setCamera('overview');
  new ResizeObserver(() => {
    const { width, height } = viewport.getBoundingClientRect();
    renderer.setSize(width, Math.max(height, 1));
    camera.aspect = width / Math.max(height, 1);
    camera.updateProjectionMatrix();
  }).observe(viewport);

  const rotation = new THREE.Matrix4();
  const forward = new THREE.Vector3();
  const up = new THREE.Vector3();
  const right = new THREE.Vector3();
  const target = new THREE.Vector3();
  const quaternion = (physics) => {
    forward.fromArray(physics.forward);
    up.fromArray(physics.up);
    right.crossVectors(up, forward).normalize();
    rotation.makeBasis(forward, right, up);
    return new THREE.Quaternion().setFromRotationMatrix(rotation);
  };
  const position = (object, before, after, alpha) => {
    object.position.fromArray(before.position).lerp(target.fromArray(after.position), alpha);
  };

  const stream = new EventSource('/state');
  stream.onmessage = (event) => {
    frames.push({ ...JSON.parse(event.data), received: performance.now() });
    if (frames.length > 8) frames.shift();
  };
  stream.onerror = () => console.warn('Match stream disconnected; retrying.');

  renderer.setAnimationLoop(() => {
    // Render slightly behind the stream, without ever accumulating a playback queue.
    const time = performance.now() - 70;
    while (frames.length > 2 && frames[1].received <= time) frames.shift();
    if (frames.length) {
      const before = frames[0];
      const after = frames[1] ?? before;
      renderTraining(after.training);
      carStates = after.cars;
      const alpha = before.episode !== after.episode ? 1 : THREE.MathUtils.clamp(
        (time - before.received) / Math.max(after.received - before.received, 1), 0, 1
      );
      position(ball, before.ball, after.ball, alpha);
      after.cars.forEach((state) => {
        if (!cars.has(state.id)) {
          const car = createCar(state.team === 0 ? 0x338fff : 0xff922b, 0, 0);
          cars.set(state.id, car);
          world.add(car);
        }
        const car = cars.get(state.id);
        const previous = before.cars.find((item) => item.id === state.id) ?? state;
        position(car, previous.physics, state.physics, alpha);
        car.quaternion.copy(quaternion(previous.physics)).slerp(quaternion(state.physics), alpha);
        car.visible = !state.demoed;
      });
      after.pads.forEach(([location, active], index) => {
        if (!pads[index]) {
          const pad = new THREE.Mesh(
            new THREE.CircleGeometry(location[2] > 72 ? 100 : 45, 16),
            new THREE.MeshBasicMaterial({ color: 0xffc857, side: THREE.DoubleSide })
          );
          pad.position.set(location[0], location[1], 4);
          pads[index] = pad;
          world.add(pad);
        }
        pads[index].visible = active;
      });
    }
    const [followedId, followed] = [...cars.entries()][follow] ?? [];
    const followedState = carStates.find((state) => state.id === followedId);
    const inputs = frames[1]?.controls?.[followedId - 1] ?? frames[0]?.controls?.[followedId - 1];
    controlsInfo.hidden = !followed || !inputs;
    if (inputs && followed) {
      controlsInfo.querySelectorAll('[data-axis]').forEach((axis) => {
        const value = Math.max(-1, Math.min(1, inputs[axis.dataset.axis]));
        axis.querySelector('i').style.left = `${50 + 50 * value}%`;
      });
      controlsInfo.querySelectorAll('[data-button]').forEach((button) => {
        button.classList.toggle('active', inputs[button.dataset.button]);
      });
    }
    boostGauge.hidden = follow === null || !followed || !Number.isFinite(followedState?.boost);
    if (!boostGauge.hidden) {
      const boost = Math.round(THREE.MathUtils.clamp(followedState.boost, 0, 100));
      boostGauge.style.setProperty('--boost', `${boost}%`);
      boostGauge.setAttribute('aria-valuenow', boost);
      boostValue.textContent = boost;
    }
    if (follow !== null && followed) {
      followed.getWorldPosition(target);
      camera.position.add(target.clone().sub(controls.target));
      controls.target.copy(target);
    }
    controls.update();
    renderer.render(scene, camera);
  });

  const arena = await new OBJLoader().loadAsync('/arena.obj');
  const surface = new THREE.MeshBasicMaterial({
    color: 0x6c9cbb, side: THREE.DoubleSide, transparent: true, opacity: 0.08, depthWrite: false
  });
  const edges = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.5 });
  const neutralEdge = new THREE.Color(0x648da9);
  const blueEdge = new THREE.Color(0x398de0);
  const orangeEdge = new THREE.Color(0xe29450);
  const edgeColor = new THREE.Color();
  // Match CARL's OBJ loader: the asset stores its X/Y axes swapped.
  const meshAxes = new THREE.Matrix4().set(0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
  arena.traverse((child) => {
    if (!child.isMesh) return;
    child.geometry.applyMatrix4(meshAxes);
    child.material = surface;
    const geometry = new THREE.EdgesGeometry(child.geometry, 25);
    const positions = geometry.getAttribute('position');
    const colors = [];
    for (let i = 0; i < positions.count; i++) {
      const y = positions.getY(i);
      const strength = 0.9 * THREE.MathUtils.smoothstep(Math.abs(y), 0, 3000);
      edgeColor.copy(neutralEdge).lerp(y < 0 ? blueEdge : orangeEdge, strength);
      colors.push(edgeColor.r, edgeColor.g, edgeColor.b);
    }
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    child.add(new THREE.LineSegments(geometry, edges));
  });
  world.add(arena);
}

start().catch((error) => {
  console.error(error);
});
