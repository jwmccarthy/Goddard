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
    roster: [],
  }
}

function setCamera(view, name) {
  const car = view.roster.find((car) => String(car.id) === name)
  view.follow = car?.id ?? null
  const direction = car ? (car.team === 0 ? 1 : -1) : 0
  if (direction) {
    view.camera.position.set(500, 450, direction * 2500)
    view.controls.target.set(0, 60, direction * 1600)
  } else {
    view.camera.position.set(9200, 10000, 11500)
    view.controls.target.set(0, 0, 0)
  }
  view.controls.update()
}

function updateRoster(view, cars) {
  if (cars.length === view.roster.length && cars.every((car, index) => {
    const previous = view.roster[index]
    return car.id === previous.id && car.team === previous.team
  })) return

  view.roster = cars.map(({ id, team }) => ({ id, team }))
  const selected = cars.some((car) => car.id === view.follow)
  if (!selected) view.follow = null

  for (const [id, car] of view.cars) {
    if (cars.some((state) => state.id === id)) continue
    view.world.remove(car)
    view.cars.delete(id)
  }
}

function bindKeyboard(view) {
  window.addEventListener('keydown', (event) => {
    if (event.target.closest('input, select, button, textarea')) return
    if (event.key === '0') setCamera(view, 'overview')
    const car = /^[1-6]$/.test(event.key)
      ? view.roster[Number(event.key) - 1]
      : null
    if (car) setCamera(view, String(car.id))
    const command = {
      r: '/reset/replay',
      k: '/reset/kickoff',
      c: '/reload/checkpoint',
    }[event.key.toLowerCase()]
    if (!command || event.repeat) return
    fetch(command, {
      method: 'POST',
    }).then((response) => {
      if (!response.ok) console.error(`Viewer command failed: ${response.status}`)
    }).catch((error) => console.error('Viewer command failed:', error))
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
    const color = state.team === 0 ? 0x338fff : 0xff922b
    if (!view.cars.has(state.id)) {
      const car = createCar(color)
      view.cars.set(state.id, car)
      view.world.add(car)
    }

    const car = view.cars.get(state.id)
    car.userData.paint.color.setHex(color)
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
    updateRoster(view, after.cars)
    updateCars(view, before, after, alpha)
    updatePads(view, after.pads)
  }

  const car = view.cars.get(view.follow)
  const index = after?.cars.findIndex((item) => item.id === view.follow) ?? -1
  const state = after?.cars[index]
  const inputs = after?.controls?.[index]
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
