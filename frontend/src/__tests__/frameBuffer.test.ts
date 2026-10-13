/** フレームの受信と描画の時刻が異なっても、表示区間と経路を保てるかを検査する。 */

import assert from 'node:assert/strict'
import { register } from 'node:module'
import { beforeEach, describe, test } from 'node:test'
import type { TestContext } from 'node:test'
import type { FrameMessage } from '../types/protocol.ts'
import { hooksUrl } from './support/tsxHooks.ts'

register(hooksUrl)
const { frameBuffer, pushFrame, resetFrameBuffer } = await import('../store/frameBuffer.ts')
const { playoutTime } = await import('../store/playout.ts')
const { computeAlpha, createPose, sampleVehicle } = await import('../scene/interpolation.ts')

function frame(simTime: number): FrameMessage {
  return {
    type: 'frame',
    tick: Math.round(simTime * 20),
    simTime,
    vehicles: [{
      id: 0, active: true, x: simTime * 10, y: 0, heading: 0, speed: 10, steer: 0,
      collided: false, reachedGoal: false, goal: [100, 0], progress: 0,
      signalViolations: 0, laneDepartures: 0, speedLimit: 0, speedViolations: 0,
    }],
    obstacles: [],
  }
}

function receiver(t: TestContext) {
  let wall = 0
  t.mock.method(performance, 'now', () => wall)
  return (simTime: number, wallMs: number, message = frame(simTime)) => {
    wall = wallMs
    pushFrame(message)
  }
}

function crossingArrival(t: TestContext): number {
  const receive = receiver(t)
  for (let k = 0; k <= 20; k++) {
    receive(k / 20, k * 50)
    computeAlpha(k * 50, false)
  }
  const clock = frameBuffer.clock
  const boundaryWall = clock.anchorWall
    + ((frameBuffer.curr!.simTime - clock.anchorSim) / clock.slope) * 1000
  receive(1.05, boundaryWall + 5)
  return boundaryWall - 5
}

describe('フレームの受信と再生', () => {
  beforeEach(resetFrameBuffer)

  test('受信時刻が表示区間の境界を越えても、描画まで区間を繰り上げない', (t) => {
    const receive = receiver(t)
    receive(0, 0)
    receive(0.05, 50)
    receive(0.1, 100)
    computeAlpha(140, false)
    const { prev, curr, displayed } = frameBuffer
    receive(0.15, 150)
    receive(0.2, 200)
    assert.equal(frameBuffer.prev, prev)
    assert.equal(frameBuffer.curr, curr)
    assert.equal(frameBuffer.displayed, displayed)
    assert.equal(frameBuffer.received, 5)
  })

  test('受信直前の rAF 時刻で補間しても係数が 0 に張り付かず、車体とカメラの位置が揃う', (t) => {
    const rafTime = crossingArrival(t)
    const expected = playoutTime(frameBuffer.clock, rafTime)
    const alpha = computeAlpha(rafTime, false)
    assert.ok(alpha > 0 && alpha < 1, `補間係数 ${alpha}`)
    const vehicle = createPose()
    const camera = createPose()
    assert.equal(sampleVehicle(0, alpha, vehicle), true)
    assert.equal(sampleVehicle(0, computeAlpha(rafTime, false), camera), true)
    assert.ok(Math.abs(vehicle.x - expected * 10) < 1e-9)
    assert.equal(camera.x, vehicle.x)
  })

  test('描画が長く止まっても最新の待ち行列だけ残し、捨てたフレームの経路も保持して復帰する', (t) => {
    const receive = receiver(t)
    receive(0, 0)
    const displayed = frameBuffer.displayed
    const route: [number, number][] = [[0, 0], [100, 0]]
    for (let k = 1; k <= 2000; k++) {
      const message = frame(k / 20)
      if (k === 1) {
        message.vehicles[0].route = route
        message.vehicles[0].routeZ = [0, 1]
      }
      receive(k / 20, k * 50, message)
      assert.ok(frameBuffer.pending.length <= 30)
    }
    assert.equal(frameBuffer.pending.length, 30)
    assert.equal(frameBuffer.pending[0].tick, 1971)
    assert.equal(frameBuffer.pending[29].tick, 2000)
    assert.equal(frameBuffer.curr!.simTime, 0)
    assert.equal(frameBuffer.prev, null)
    assert.equal(frameBuffer.displayed, displayed)
    assert.deepEqual(frameBuffer.routes.get(0), route)
    assert.deepEqual(frameBuffer.routeHeights.get(0), [0, 1])

    const pose = createPose()
    let previous = 0
    for (const wall of [100_000, 100_016, 100_032, 100_200]) {
      assert.equal(sampleVehicle(0, computeAlpha(wall, false), pose), true)
      assert.ok(Math.abs(pose.x - playoutTime(frameBuffer.clock, wall) * 10) < 1e-9)
      assert.ok(pose.x >= previous && pose.x <= 1000)
      previous = pose.x
    }
    assert.equal(frameBuffer.pending.length, 0)
  })

  test('一時停止中は最新を表示し、次の受信から時計を張り直す', (t) => {
    const receive = receiver(t)
    for (let k = 0; k <= 10; k++) receive(k / 20, k * 50)
    assert.equal(computeAlpha(500, true), 1)
    assert.equal(frameBuffer.curr!.simTime, 0.5)
    assert.equal(frameBuffer.pending.length, 0)
    assert.equal(frameBuffer.clock.ready, false)
    receive(0.55, 550)
    assert.equal(frameBuffer.prev, null)
    assert.equal(frameBuffer.curr!.simTime, 0.55)
    assert.equal(frameBuffer.clock.ready, true)
  })

  test('同時刻の受信は積まず、simTime が戻ったら古い待ち行列を捨てる', (t) => {
    const receive = receiver(t)
    receive(1, 0)
    receive(1.05, 50)
    receive(1.05, 60)
    assert.equal(frameBuffer.pending.length, 1)
    receive(0, 100)
    assert.equal(frameBuffer.prev, null)
    assert.equal(frameBuffer.curr!.simTime, 0)
    assert.equal(frameBuffer.pending.length, 0)
  })
})
