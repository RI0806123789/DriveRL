import assert from 'node:assert/strict'
import { test } from 'node:test'
import * as THREE from 'three'
import { createTrafficSignGeometry, signArrowAngles, trafficSignOutline } from '../scene/trafficSignVisual.ts'
import { signBoardFacing } from '../scene/signGeometry.ts'
import type { SignKind } from '../types/protocol.ts'
import { detectionLabel } from '../scene/detectionLabels.ts'

const kinds: SignKind[] = ['stop', 'crosswalk', 'one_way', 'mandatory_direction', 'no_parking', 'no_stopping']
test('新標識の面は進入車両に正対し、寸法と UV が有効', () => {
  for (const kind of kinds) {
    const face = createTrafficSignGeometry(kind, true)
    const board = createTrafficSignGeometry(kind, false)
    face.computeBoundingBox()
    board.computeBoundingBox()
    assert.ok(face.boundingBox!.min.x > board.boundingBox!.max.x)
    const uv = face.getAttribute('uv')
    for (const value of uv.array) assert.ok(value >= 0 && value <= 1)
    const normal = new THREE.Vector3().fromBufferAttribute(face.getAttribute('normal'), 0)
    for (const heading of [0, Math.PI / 2, Math.PI, -Math.PI / 2]) {
      const rotated = normal.clone().applyAxisAngle(new THREE.Vector3(0, 1, 0), signBoardFacing({ x: 0, y: 0, heading, speedLimit: 0 }))
      assert.ok(rotated.dot(new THREE.Vector3(Math.cos(heading), 0, -Math.sin(heading))) < -0.999)
    }
    assert.ok(Math.abs(board.boundingBox!.max.y - 0.3 * (kind === 'one_way' ? 0.5 : 1)) < 1e-6)
    face.dispose()
    board.dispose()
  }
  assert.equal(trafficSignOutline('stop')!.length, 3)
  assert.equal(trafficSignOutline('crosswalk')!.length, 4)
})

test('指定方向の矢印は許可する方向だけを描く', () => {
  assert.deepEqual(signArrowAngles('left'), [-Math.PI / 2])
  assert.deepEqual(signArrowAngles('right'), [Math.PI / 2])
  assert.deepEqual(signArrowAngles('straight'), [0])
  assert.deepEqual(signArrowAngles('left_or_right'), [-Math.PI / 2, Math.PI / 2])
  assert.deepEqual(signArrowAngles('left_or_straight'), [-Math.PI / 2, 0])
  assert.deepEqual(signArrowAngles('right_or_straight'), [Math.PI / 2, 0])
})

test('検出クラスを日本語で区別する', () => {
  const names = ['一時停止', '横断歩道', '一方通行', '指定方向外進行禁止', '駐車禁止', '駐停車禁止']
  names.forEach((name, i) => assert.equal(detectionLabel({ cls: i + 6, conf: 1, box: [0, 0, 1, 1] }), name))
})
