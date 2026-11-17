import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';
import { OBJLoader } from './vendor/OBJLoader.js';

const viewport = document.querySelector('#viewport');
const localPlayback = document.body.dataset.playback === 'local';
const boostGauge = document.querySelector('#boost-gauge');
const boostValue = document.querySelector('#boost-value');
const trainingInfo = document.querySelector('#training-info');
const trainingStages = document.querySelector('#training-stages');
const controlsInfo = document.querySelector('#controls-info');
const expertInfo = document.querySelector('#expert-info');
const expertClip = document.querySelector('#expert-clip');
const expertSeek = document.querySelector('#expert-seek');
const expertSpeed = document.querySelector('#expert-speed');
let expertMode = false;
let expertListRequested = false;
let expertSeeking = false;
let lastExpert = null;

async function expertControl(action, value = null) {
  try {
    const response = await fetch('/expert/control', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action, value }),
    });
    if (!response.ok) throw new Error(`Playback command failed: ${response.status}`);
  } catch (error) {
    document.querySelector('#expert-error').textContent = error.message;
  }
}

for (const [id, action, value] of [
  ['previous', 'previous', null], ['next', 'next', null], ['restart', 'restart', null],
  ['play', 'toggle', null], ['back', 'step', -1], ['forward', 'step', 1],
]) {
  document.querySelector(`#expert-${id}`)?.addEventListener('click', () => expertControl(action, value));
}
expertClip?.addEventListener('change', () => expertControl('select', Number(expertClip.value)));
expertSpeed?.addEventListener('change', () => expertControl('speed', Number(expertSpeed.value)));
document.querySelector('#expert-loop')?.addEventListener('change', (event) => expertControl('loop', event.target.checked));
document.querySelector('#expert-core')?.addEventListener('click', () => {
  if (lastExpert && lastExpert.core_start !== null) {
    expertControl('seek', lastExpert.core_start - (lastExpert.source_row - lastExpert.frame));
  }
});
expertSeek?.addEventListener('pointerdown', () => { expertSeeking = true; expertControl('pause'); });
expertSeek?.addEventListener('input', () => { expertSeeking = true; });
expertSeek?.addEventListener('change', () => {
  expertControl('seek', Number(expertSeek.value));
  expertSeeking = false;
});
expertSeek?.addEventListener('blur', () => { expertSeeking = false; });

function renderExpert(expert) {
  if (!expertInfo) return;
  if (expert === lastExpert) return;
  lastExpert = expert;
  expertInfo.hidden = !expert;
  expertMode = Boolean(expert);
  if (!expert) return;
  if (!expertListRequested) {
    expertListRequested = true;
    fetch('/expert/clips').then((response) => {
      if (!response.ok) throw new Error('Could not load demonstration list');
      return response.json();
    }).then((clips) => {
      const options = document.createDocumentFragment();
      clips.forEach((clip) => {
        const option = document.createElement('option');
        option.value = clip.index;
        option.textContent = clip.label;
        options.append(option);
      });
      expertClip.replaceChildren(options);
      expertClip.value = String(expert.index);
    }).catch((error) => { document.querySelector('#expert-error').textContent = error.message; });
  }
  if (document.activeElement !== expertClip) expertClip.value = String(expert.index);
  expertSeek.max = Math.max(0, expert.frames - 1);
  if (!expertSeeking) expertSeek.value = expert.frame;
  if (![...expertSpeed.options].some((option) => option.value === String(expert.speed))) {
    const option = document.createElement('option');
    option.value = String(expert.speed);
    option.textContent = `${expert.speed}×`;
    expertSpeed.append(option);
  }
  expertSpeed.value = String(expert.speed);
  document.querySelector('#expert-play').textContent = expert.paused ? 'Play' : 'Pause';
  document.querySelector('#expert-core').disabled = expert.core_start === null;
  document.querySelector('#expert-loop').checked = expert.loop;
  document.querySelector('#expert-time').textContent = `${expert.index}/${expert.count} · ${expert.time.toFixed(2)} / ${expert.duration.toFixed(2)} s · source row ${expert.source_row} · ${expert.cadence} physics ticks/frame`;
  const bounds = expert.core_start === null ? '' : ` · core rows ${expert.core_start}–${expert.core_stop - 1}`;
  document.querySelector('#expert-phase').textContent = `${expert.phase}${bounds}${expert.touch ? ' · EGO TOUCH' : ''}${expert.boosting ? ' · BOOSTING' : ''}${expert.flagged ? ' · FLAGGED SOURCE ROW' : ''}`;
  document.querySelector('#expert-metrics').textContent = `Ball distance ${expert.distance.toFixed(0)} uu · car/ball height ${expert.car_height.toFixed(0)}/${expert.ball_height.toFixed(0)} uu · touch-flag rows so far ${expert.touch_flags}`;
  document.querySelector('#expert-source').textContent = expert.source;
  const scores = Object.entries(expert.scores).map(([label, values]) => `${label} P(expert), ego/global: ${values.map((value) => `${(100 * value).toFixed(1)}%`).join(' / ')}`).join(' · ');
  document.querySelector('#expert-scores').textContent = scores ? `Stored audit means — ${scores}` : '';
  document.querySelector('#expert-error').textContent = expert.error ?? '';
}

let trainingLayout = '';

function renderTraining(training) {
  if (!trainingInfo) return;
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
  car.userData.paint = paint;
  const body = new THREE.Mesh(new THREE.BoxGeometry(120.5, 86.7, 38.7), paint);
  body.position.set(13.9, 0, 20.75);
  car.add(body);

  // Car-local indicators: warm exhaust for boost, a cyan roof arrow for held jump.
  // These follow recorded state, so pausing or scrubbing never leaves a stale flash.
  const boost = new THREE.Group();
  const glow = new THREE.MeshBasicMaterial({ color: 0xffbe50, toneMapped: false });
  for (const y of [-23, 23]) {
    const exhaust = new THREE.Mesh(new THREE.BoxGeometry(32, 16, 16), glow);
    exhaust.position.set(-58, y, 19);
    boost.add(exhaust);
  }
  boost.visible = false;
  car.add(boost);
  const jump = new THREE.Group();
  const cyan = new THREE.MeshBasicMaterial({ color: 0x62f0ff, toneMapped: false });
  const stem = new THREE.Mesh(new THREE.CylinderGeometry(3, 3, 18, 8), cyan);
  stem.rotation.x = Math.PI / 2;
  stem.position.set(14, 0, 51);
  const arrow = new THREE.Mesh(new THREE.ConeGeometry(11, 18, 4), cyan);
  arrow.rotation.x = Math.PI / 2;
  arrow.position.set(14, 0, 69);
  jump.add(stem, arrow);
  jump.visible = false;
  car.add(jump);
  car.userData.boost = boost;
  car.userData.jump = jump;

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
  renderer.domElement.title = 'Orange exhaust: boosting · Cyan roof arrow: jump held';
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
  let customColors = {};

  function setCamera(name) {
    follow = typeof name === 'number' ? name - 1 : name === 'overview' ? null : name === 'blue' ? 0 : 1;
    if (follow === null) boostGauge.hidden = true;
    const direction = typeof name === 'number' ? 1 : { blue: 1, orange: -1 }[name];
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
    if (localPlayback) return; // The labeler owns keyboard shortcuts and its camera dropdown.
    if (event.target.closest('input, select, button, textarea')) return;
    if (expertMode && ['Space', 'ArrowLeft', 'ArrowRight'].includes(event.code)) {
      event.preventDefault();
      if (!event.repeat || event.code !== 'Space') {
        expertControl(event.code === 'Space' ? 'toggle' : 'step', event.code === 'ArrowLeft' ? -1 : event.code === 'ArrowRight' ? 1 : null);
      }
      return;
    }
    const name = { '0': 'overview', '1': 'blue', '2': 'orange' }[event.key];
    if (name) setCamera(name);
    const reset = { r: 'replay', k: 'kickoff', p: 'pack', n: 'next', b: 'previous' }[event.key.toLowerCase()];
    if (expertMode && ['kickoff', 'pack'].includes(reset)) return;
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

  if (localPlayback) {
    window.addEventListener('goddard:frame', (event) => {
      frames.length = 0;
      frames.push({ ...event.detail, received: performance.now() });
    });
    window.addEventListener('goddard:camera', (event) => setCamera(event.detail));
    window.addEventListener('goddard:colors', (event) => {
      customColors = event.detail.car_colors ?? {};
      ball.material.color.set(event.detail.ball_color ?? '#f0eee4');
    });
  } else {
    const stream = new EventSource('/state');
    stream.onmessage = (event) => {
      const state = JSON.parse(event.data);
      if (state.expert && frames.at(-1)?.episode !== state.episode) frames.length = 0;
      frames.push({ ...state, received: performance.now() });
      if (frames.length > 8) frames.shift();
    };
    stream.onerror = () => console.warn('Match stream disconnected; retrying.');
  }

  renderer.setAnimationLoop(() => {
    // Render slightly behind the stream, without ever accumulating a playback queue.
    const time = performance.now() - 70;
    while (frames.length > 2 && frames[1].received <= time) frames.shift();
    if (frames.length) {
      const before = frames[0];
      const after = frames[1] ?? before;
      const firstExpert = Boolean(after.expert) && !expertMode;
      renderExpert(after.expert);
      if (firstExpert) setCamera('blue');
      renderTraining(after.training);
      carStates = after.cars;
      const alpha = before.episode !== after.episode ? 1 : THREE.MathUtils.clamp(
        (time - before.received) / Math.max(after.received - before.received, 1), 0, 1
      );
      position(ball, before.ball, after.ball, alpha);
      if (after.expert || localPlayback) {
        const ids = new Set(after.cars.map((state) => state.id));
        for (const [id, car] of cars) {
          if (ids.has(id)) continue;
          world.remove(car);
          const materials = new Set();
          car.traverse((object) => {
            if (object.isMesh) { object.geometry.dispose(); materials.add(object.material); }
          });
          materials.forEach((material) => material.dispose());
          cars.delete(id);
        }
      }
      after.cars.forEach((state) => {
        if (!cars.has(state.id)) {
          const car = createCar(state.team === 0 ? 0x338fff : 0xff922b, 0, 0);
          cars.set(state.id, car);
          world.add(car);
        }
        const car = cars.get(state.id);
        if (localPlayback) car.userData.paint.color.set(customColors[state.id] ?? (state.team === 0 ? '#338fff' : '#ff922b'));
        const previous = before.cars.find((item) => item.id === state.id) ?? state;
        position(car, previous.physics, state.physics, alpha);
        car.quaternion.copy(quaternion(previous.physics)).slerp(quaternion(state.physics), alpha);
        car.visible = !state.demoed;
        const signal = alpha < 1 ? previous : state;
        const inputs = (alpha < 1 ? before : after).controls?.[state.id - 1];
        const boosting = signal.boosting ?? (Boolean(inputs?.boost) && signal.boost > 0);
        car.userData.boost.visible = Boolean(boosting);
        car.userData.jump.visible = Boolean(signal.jumping ?? inputs?.jump);
        car.userData.paint.emissive.setHex(boosting ? 0x9a510a : 0x000000);
        car.userData.paint.emissiveIntensity = boosting ? 0.35 : 0;
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
  window.dispatchEvent(new CustomEvent('goddard:viewer-error', { detail: error.message }));
});
