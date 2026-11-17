import * as THREE from 'three'
import { OrbitControls } from './vendor/OrbitControls.js'
import { createCar, loadArena } from './scene.js'
import { updateHud } from './hud.js'

const viewport = document.querySelector('#viewport')

function createView() {
  const renderer = new THREE.WebGLRenderer({ antialias: true })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5))
  viewport.append(renderer.domElement)
  renderer.domElement.title = 'Orange exhaust: boosting · Cyan roof arrow: jump held'

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x0b1220)
  scene.add(new THREE.HemisphereLight(0xe2efff, 0x344154, 3))
  const world = new THREE.Group()
  world.rotation.x = -Math.PI / 2
  scene.add(world)
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(8192, 10240),
    new THREE.MeshStandardMaterial({ color: 0x173c3b })
  )
  floor.position.z = -2
  world.add(floor)

  const ball = new THREE.Mesh(
    new THREE.SphereGeometry(91.25, 24, 16),
    new THREE.MeshStandardMaterial({ color: 0xf0eee4 })
  )
  ball.position.set(0, 0, 93.15)
  world.add(ball)
  const camera = new THREE.PerspectiveCamera(55, 1, 5, 60000)
  const controls = new OrbitControls(camera, renderer.domElement)
  controls.enableDamping = true
  controls.minDistance = 120
  controls.maxDistance = 26000
  controls.maxPolarAngle = Math.PI / 2 - 0.02
  controls.zoomToCursor = true

  new ResizeObserver(() => {
    const { width, height } = viewport.getBoundingClientRect()
    renderer.setSize(width, Math.max(height, 1))
    camera.aspect = width / Math.max(height, 1)
    camera.updateProjectionMatrix()
  }).observe(viewport)

  return {
    renderer,
    scene,
    world,
    ball,
    camera,
    controls,
    cars:   new Map(),
    pads:   [],
    follow: null,
  }
}

function setCamera(view, name) {
  view.follow = name === 'overview' ? null : name === 'blue' ? 0 : 1
  const direction = {
    blue:   1,
    orange: -1,
  }[name]
  if (direction) {
    view.camera.position.set(500, 450, direction * 2500)
    view.controls.target.set(0, 60, direction * 1600)
  } else {
    view.camera.position.set(9200, 10000, 11500)
    view.controls.target.set(0, 0, 0)
  }
  view.controls.update()
}

function bindKeyboard(view) {
  window.addEventListener('keydown', (event) => {
    if (event.target.closest('input, select, button, textarea')) return
    const name = {
      '0': 'overview',
      '1': 'blue',
      '2': 'orange',
    }[event.key]
    if (name) setCamera(view, name)
    const reset = {
      r: 'replay',
      k: 'kickoff',
    }[event.key.toLowerCase()]
    if (!reset || event.repeat) return
    fetch(`/reset/${reset}`, {
      method: 'POST',
    }).then((response) => {
      if (!response.ok) console.error(`Reset failed: ${response.status}`)
    }).catch((error) => console.error('Reset failed:', error))
  })
}

const rotation = new THREE.Matrix4()
const forward = new THREE.Vector3()
const up = new THREE.Vector3()
const right = new THREE.Vector3()
const target = new THREE.Vector3()

function quaternion(physics) {
  forward.fromArray(physics.forward)
  up.fromArray(physics.up)
  right.crossVectors(up, forward).normalize()
  rotation.makeBasis(forward, right, up)
  return new THREE.Quaternion().setFromRotationMatrix(rotation)
}

function position(object, before, after, alpha) {
  object.position
    .fromArray(before.position)
    .lerp(target.fromArray(after.position), alpha)
}

function updateCars(view, before, after, alpha) {
  after.cars.forEach((state) => {
    if (!view.cars.has(state.id)) {
      const car = createCar(state.team === 0 ? 0x338fff : 0xff922b)
      view.cars.set(state.id, car)
      view.world.add(car)
    }

    const car = view.cars.get(state.id)
    const previous = before.cars.find((item) => item.id === state.id) ?? state
    position(car, previous.physics, state.physics, alpha)
    car.quaternion
      .copy(quaternion(previous.physics))
      .slerp(quaternion(state.physics), alpha)
    car.visible = !state.demoed
    const signal = alpha < 1 ? previous : state
    const inputs = (alpha < 1 ? before : after).controls?.[state.id - 1]
    const boostHeld = Boolean(inputs?.boost) && signal.boost > 0
    const boosting = signal.boosting ?? boostHeld
    car.userData.boost.visible = Boolean(boosting)
    car.userData.jump.visible = Boolean(inputs?.jump)
    car.userData.paint.emissive.setHex(boosting ? 0x9a510a : 0x000000)
    car.userData.paint.emissiveIntensity = boosting ? 0.35 : 0
  })
}

function updatePads(view, states) {
  states.forEach(([location, active], index) => {
    if (!view.pads[index]) {
      const pad = new THREE.Mesh(
        new THREE.CircleGeometry(location[2] > 72 ? 100 : 45, 16),
        new THREE.MeshBasicMaterial({
          color: 0xffc857,
          side:  THREE.DoubleSide
        })
      )
      pad.position.set(location[0], location[1], 4)
      view.pads[index] = pad
      view.world.add(pad)
    }
    view.pads[index].visible = active
  })
}

function render(view, frames) {
  const time = performance.now() - 70
  while (frames.length > 2 && frames[1].received <= time) frames.shift()
  const before = frames[0]
  const after = frames[1] ?? before

  if (after) {
    const elapsed = time - before.received
    const duration = Math.max(after.received - before.received, 1)
    const alpha = before.episode !== after.episode
      ? 1
      : THREE.MathUtils.clamp(elapsed / duration, 0, 1)

    position(view.ball, before.ball, after.ball, alpha)
    updateCars(view, before, after, alpha)
    updatePads(view, after.pads)
  }

  const [id, car] = [...view.cars.entries()][view.follow] ?? []
  const state = after?.cars.find((item) => item.id === id)
  const inputs = after?.controls?.[id - 1]
  updateHud(car, state, inputs)
  if (car) {
    car.getWorldPosition(target)
    view.camera.position.add(target.clone().sub(view.controls.target))
    view.controls.target.copy(target)
  }
  view.controls.update()
  view.renderer.render(view.scene, view.camera)
}

async function start() {
  const view = createView()
  bindKeyboard(view)
  setCamera(view, 'overview')
  const frames = []
  const stream = new EventSource('/state')
  stream.onmessage = (event) => {
    frames.push({ ...JSON.parse(event.data), received: performance.now() })
    if (frames.length > 8) frames.shift()
  }
  stream.onerror = () => console.warn('Match stream disconnected. Retrying.')
  view.renderer.setAnimationLoop(() => render(view, frames))
  view.world.add(await loadArena())
}

start().catch(console.error)
