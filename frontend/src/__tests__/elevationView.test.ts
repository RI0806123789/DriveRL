/** 立体の地図の路面モデル（scene/elevationView.ts）とカメラ・車体の姿勢の単体テスト。 */

import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { describe, test } from 'node:test'
import * as THREE from 'three'

import type { MapEdge, MapMessage } from '../types/protocol.ts'
import {
  DECK_THICKNESS_M,
  LEVEL_GAP_M,
  STRUCTURE_MARGIN_M,
  TUNNEL_CLEARANCE_M,
  TUNNEL_LIGHT_FACTOR,
  attitudeAxes,
  attitudeLocalToThree,
  buildElevationModel,
  pickSurface,
  surfaceZAt,
  terrainZ,
} from '../scene/elevationView.ts'
import { DRIVER_EYE_LOCAL, driverPlacement, followPlacement, surroundPlacement, SURROUND_CAMERAS } from '../scene/cameraMath.ts'
import { composeVehicleMatrix, createTransformScratch } from '../scene/vehicleGeometry.ts'
import { createTiltState, resetTilt, stepTilt } from '../scene/vehicleMotion.ts'

const EPS = 1e-9

function edge(id: number, polyline: [number, number][], extra: Partial<MapEdge> = {}): MapEdge {
  return { id, u: id * 2, v: id * 2 + 1, lanes: 2, width: 6.5, oneway: false, speedLimit: 11.1, polyline, ...extra }
}

function elevatedMap(): MapMessage {
  return {
    type: 'map',
    presetId: 'lab',
    name: 'lab',
    bounds: { minX: -50, maxX: 50, minY: -50, maxY: 50 },
    nodes: [],
    edges: [
      edge(0, [[-40, 0], [40, 0]], { elevation: [0, 8], structure: 'ground' }),
      edge(1, [[0, -40], [0, 40]], { elevation: [6.5, 6.5], structure: 'bridge', crossSlope: 0.05 }),
    ],
    buildings: [],
    elevated: true,
    terrain: { originX: -50, originY: -50, cellSize: 50, width: 3, height: 3, z: [0, 0, 0, 0, 2, 0, 0, 0, 0], source: 'synthetic' },
  }
}

describe('路面モデル', () => {
  test('平面の地図では作らない', () => {
    const flat = { ...elevatedMap(), elevated: undefined, terrain: undefined }
    assert.equal(buildElevationModel(flat), null)
    assert.equal(surfaceZAt(null, 10, 10), 0)
  })

  test('地形は双一次で補間する', () => {
    const model = buildElevationModel(elevatedMap())
    assert.ok(model)
    assert.ok(Math.abs(terrainZ(model.terrain, 0, 0) - 2) < EPS)
    assert.ok(Math.abs(terrainZ(model.terrain, 25, 0) - 1) < EPS)
  })

  test('重なった道路は手がかりの高さに近い方を選び、無ければ地面に近い方', () => {
    const model = buildElevationModel(elevatedMap())
    assert.ok(model)
    // 交点 (0, 0)：東西の道は 4m、南北の橋は 6.5m（横断勾配 5% で左右へ ±）
    assert.ok(Math.abs(surfaceZAt(model, 0, 0, 6.5) - 6.5) < 0.01)
    assert.ok(Math.abs(surfaceZAt(model, 0, 0, 0) - 2) < 0.01)
    assert.ok(Math.abs(surfaceZAt(model, 1, 0, 6.5) - (6.5 - 1 * 0.05)) < 0.01)
  })

  test('視線は上から見える路面（橋桁の上）で拾う', () => {
    const model = buildElevationModel(elevatedMap())
    assert.ok(model)
    const hit = pickSurface(model, { x: 0, y: 40, z: 0 }, { x: 0, y: -1, z: 0 })
    assert.ok(hit)
    assert.ok(Math.abs(hit.z - 6.5) < 0.05, `${hit.z}`)
  })

  test('バックエンドの定数と同じ値', () => {
    const configPy = readFileSync(new URL('../../../backend/app/config.py', import.meta.url), 'utf8')
    const elevationPy = readFileSync(new URL('../../../backend/app/map/elevation.py', import.meta.url), 'utf8')
    const value = (source: string, name: string) => Number(new RegExp(`^${name} = ([0-9.]+)`, 'm').exec(source)?.[1])
    assert.equal(value(configPy, 'TUNNEL_LIGHT_FACTOR'), TUNNEL_LIGHT_FACTOR)
    assert.equal(value(elevationPy, 'DECK_THICKNESS_M'), DECK_THICKNESS_M)
    assert.equal(value(elevationPy, 'TUNNEL_CLEARANCE_M'), TUNNEL_CLEARANCE_M)
    assert.equal(value(elevationPy, 'STRUCTURE_MARGIN_M'), STRUCTURE_MARGIN_M)
    assert.equal(value(elevationPy, 'LEVEL_GAP_M'), LEVEL_GAP_M)
  })
})

describe('路面の姿勢', () => {
  test('ピッチ・ロールが 0 なら方位の回転と同じ', () => {
    for (const h of [0, 0.7, -2.4, Math.PI]) {
      const a = attitudeAxes(h, 0, 0)
      const m = new THREE.Matrix4().makeRotationY(h)
      const e = m.elements
      const expected = [e[0], e[1], e[2], e[4], e[5], e[6], e[8], e[9], e[10]]
      for (let i = 0; i < 9; i++) assert.ok(Math.abs(a[i] - expected[i]) < 1e-12, `${h} ${i}`)
    }
  })

  test('機首上げのピッチで前が上を向き、左が上がるロールで右が下がる', () => {
    const a = attitudeAxes(0, 0.1, 0)
    assert.ok(a[1] > 0, '前の高さ成分')
    const b = attitudeAxes(0, 0, 0.1)
    assert.ok(b[7] < 0, '右の高さ成分')
  })

  test('車体の行列と車両ローカルの点の写し方が一致する', () => {
    const scratch = createTransformScratch()
    const m = new THREE.Matrix4()
    const pose = { x: 12, y: -30, z: 4.5, heading: 0.9, pitch: 0.07, roll: -0.04 }
    composeVehicleMatrix(scratch, pose.x, pose.y, pose.heading, m, pose.z, pose.pitch, pose.roll)
    const local: [number, number, number] = [0.35, 1.22, 0.36]
    const v = new THREE.Vector3(...local).applyMatrix4(m)
    const w = attitudeLocalToThree(local, pose.x, pose.y, pose.z, pose.heading, pose.pitch, pose.roll)
    assert.ok(Math.abs(v.x - w.x) < 1e-6 && Math.abs(v.y - w.y) < 1e-6 && Math.abs(v.z - w.z) < 1e-6)
  })

  test('平面の地図のカメラは従来の置き方のまま', () => {
    const flat = { x: 5, y: 7, heading: 1.1, z: 0, pitch: 0, roll: 0 }
    const placed = driverPlacement(flat, DRIVER_EYE_LOCAL)
    assert.deepEqual(placed.up, { x: 0, y: 1, z: 0 })
    const tilted = driverPlacement({ ...flat, z: 3, pitch: 0.08, roll: 0 }, DRIVER_EYE_LOCAL)
    // 上り坂では、目は路面の高さぶん上がり、視線も上を向く
    assert.ok(tilted.eye.y > placed.eye.y + 2.9)
    assert.ok(tilted.look.y - tilted.eye.y > placed.look.y - placed.eye.y)
    assert.ok(tilted.up.y < 1)
    const follow = followPlacement({ ...flat, z: 3 })
    assert.ok(Math.abs(follow.eye.y - followPlacement(flat).eye.y - 3) < EPS)
    for (const cam of SURROUND_CAMERAS) {
      const a = surroundPlacement(cam, flat)
      const b = surroundPlacement(cam, { ...flat, z: 1e-12, pitch: 0, roll: 0 })
      assert.ok(Math.abs(a.eye.x - b.eye.x) < 1e-6 && Math.abs(a.look.z - b.look.z) < 1e-6, cam.key)
    }
  })

  test('下り坂を惰性で走るときは、加速として沈み込まない（路面の傾きと二重に傾けない）', () => {
    const coasting = createTiltState()
    const g = 9.81
    const pitch = -0.08
    let speed = 10
    resetTilt(coasting, speed)
    for (let i = 0; i < 120; i++) {
      speed += -g * Math.sin(pitch) / 60
      stepTilt(coasting, speed, 0, 1 / 60, pitch, 0)
    }
    assert.ok(Math.abs(coasting.pitch) < 1e-3, `${coasting.pitch}`)
    const naive = createTiltState()
    speed = 10
    resetTilt(naive, speed)
    for (let i = 0; i < 120; i++) {
      speed += -g * Math.sin(pitch) / 60
      stepTilt(naive, speed, 0, 1 / 60)
    }
    assert.ok(naive.pitch > 0.002, '路面を見ないと加速と取り違える')
  })
})
