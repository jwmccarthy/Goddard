import * as THREE from 'three'
import { OBJLoader } from './vendor/OBJLoader.js'

function indicators(car) {
  const boost = new THREE.Group()
  const glow = new THREE.MeshBasicMaterial({
    color:      0xffbe50,
    toneMapped: false,
  })
  for (const y of [-23, 23]) {
    const exhaust = new THREE.Mesh(new THREE.BoxGeometry(32, 16, 16), glow)
    exhaust.position.set(-58, y, 19)
    boost.add(exhaust)
  }
  boost.visible = false
  car.add(boost)

  const jump = new THREE.Group()
  const cyan = new THREE.MeshBasicMaterial({
    color:      0x62f0ff,
    toneMapped: false,
  })
  const stem = new THREE.Mesh(new THREE.CylinderGeometry(3, 3, 18, 8), cyan)
  stem.rotation.x = Math.PI / 2
  stem.position.set(14, 0, 51)
  const arrow = new THREE.Mesh(new THREE.ConeGeometry(11, 18, 4), cyan)
  arrow.rotation.x = Math.PI / 2
  arrow.position.set(14, 0, 69)
  jump.add(stem, arrow)
  jump.visible = false
  car.add(jump)
  car.userData.boost = boost
  car.userData.jump = jump
}

export function createCar(color) {
  const car = new THREE.Group()
  const paint = new THREE.MeshStandardMaterial({ color })
  car.userData.paint = paint
  const body = new THREE.Mesh(
    new THREE.BoxGeometry(120.5, 86.7, 38.7),
    paint
  )
  body.position.set(13.9, 0, 20.75)
  car.add(body)
  indicators(car)

  const tires = new THREE.MeshStandardMaterial({ color: 0x15191e })
  // Axle position, half-width, and wheel radius.
  const axles = [
    [ 51.25, 25.9, 12.5],
    [-33.75, 29.5, 15.0],
  ]

  for (const [x, width, radius] of axles) {
    for (const side of [-1, 1]) {
      const wheel = new THREE.Mesh(
        new THREE.CylinderGeometry(radius, radius, 12, 16),
        tires
      )
      wheel.position.set(x, side * width, radius - 17)
      car.add(wheel)
    }
  }

  car.position.z = 17

  return car
}

function arenaEdges(mesh, material) {
  const geometry = new THREE.EdgesGeometry(mesh.geometry, 25)
  const positions = geometry.getAttribute('position')
  const neutral = new THREE.Color(0x648da9)
  const blue = new THREE.Color(0x398de0)
  const orange = new THREE.Color(0xe29450)
  const color = new THREE.Color()
  const colors = []

  for (let i = 0; i < positions.count; i++) {
    const y = positions.getY(i)
    const strength = 0.9 * THREE.MathUtils.smoothstep(Math.abs(y), 0, 3000)
    color.copy(neutral).lerp(y < 0 ? blue : orange, strength)
    colors.push(color.r, color.g, color.b)
  }

  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3))

  return new THREE.LineSegments(geometry, material)
}

export async function loadArena() {
  const arena = await new OBJLoader().loadAsync('/arena.obj')
  const surface = new THREE.MeshBasicMaterial({
    color:       0x6c9cbb,
    side:        THREE.DoubleSide,
    transparent: true,
    opacity:     0.08,
    depthWrite:  false,
  })
  const edges = new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent:  true,
    opacity:      0.5,
  })
  // Swap arena x/y axes into the simulator coordinate system.
  const axes = new THREE.Matrix4().set(
    0, 1, 0, 0,
    1, 0, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1
  )
  arena.traverse((child) => {
    if (!child.isMesh) return
    child.geometry.applyMatrix4(axes)
    child.material = surface
    child.add(arenaEdges(child, edges))
  })

  return arena
}
