/** 実用モードの徒歩キャラクター（決定 6）。 */

import { useEffect, useMemo, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'

import { send } from '../store/connection'
import { frameBuffer } from '../store/frameBuffer'
import {
  pedestrian,
  placePedestrian,
  releasePedestrianKeys,
  resetPedestrian,
} from '../store/pedestrian'
import { useSimStore } from '../store/simStore'
import { isBoardablePhase, isRidingPhase } from '../types/protocol'
import type { Vec2 } from '../types/protocol'
import { computeAlpha, createPose, sampleVehicle } from './interpolation'
import {
  aimedVehicle,
  alightPosition,
  buildBuildingIndex,
  resolveMove,
  touchesBuilding,
} from './pedestrianCollision'
import type { BuildingIndex, VehicleBlocker } from './pedestrianCollision'
import {
  LIMB_JOINTS,
  WALK_SPEED_MPS,
  composeLimbMatrix,
  composePedestrianMatrix,
  createPedestrianScratch,
  limbSwing,
  makeArmGeometry,
  makeHeadGeometry,
  makeHipGeometry,
  makeLegGeometry,
  makeTorsoGeometry,
  swingFor,
} from './pedestrianGeometry'
import { usePalette } from './usePalette'

/** マウス感度 [rad/px] と、見上げ・見下ろしの限界 [rad] */
const LOOK_SENSITIVITY = 0.0022
const PITCH_LIMIT = 1.35

/** 歩数の刻み。速く歩くほど手足が速く振れる */
const STRIDE_PER_METER = 2.0

/** 乗車地点で待つとき、道路中心から歩道側へ寄る距離 [m] */
const CURB_OFFSET_M = 3.2

/** 実用モードに入ったとき、最寄り車両の後方に立つ距離 [m] */
const SPAWN_BEHIND_M = 9

/**
 * 位置をサーバーへ知らせる間隔 [ms]。**60fps で送らないこと**（frameBuffer と同じ作法で、
 * 20Hz のサーバーが読むのは 1 ステップに 1 回だけ）。
 */
const POSE_REPORT_MS = 100

function isTypingTarget(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null
  return !!el && /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName)
}

/** いま画面に出ているアクティブ車両を、当たり判定用の形で集める */
function collectBlockers(out: VehicleBlocker[], pose: ReturnType<typeof createPose>): VehicleBlocker[] {
  out.length = 0
  const curr = frameBuffer.curr
  if (!curr) return out
  const alpha = computeAlpha(performance.now(), useSimStore.getState().status.renderPaused)
  for (const v of curr.vehicles) {
    if (!v.active) continue
    if (!sampleVehicle(v.id, alpha, pose)) continue
    out.push({ id: v.id, x: pose.x, y: pose.y, heading: pose.heading })
  }
  return out
}

/** 経路の終点での進行方向 [rad]。点が足りなければ 0 */
function routeHeadingAtEnd(route: Vec2[] | undefined): number {
  if (!route || route.length < 2) return 0
  const a = route[route.length - 2]
  const b = route[route.length - 1]
  return Math.atan2(b[1] - a[1], b[0] - a[0])
}

/**
 * 乗車地点の歩道へ立たせる。**まだ街に立っていないときだけ呼ぶこと。**
 * 乗車地点は利用者の現在地なので、呼んだ瞬間に立たせ直すと自分がワープする。
 */
function placeAtPickup(taxi: { pickup: Vec2 | null; route?: Vec2[] }, index: BuildingIndex | null): void {
  if (!taxi.pickup) return
  const dir = routeHeadingAtEnd(taxi.route)
  const curb = {
    x: taxi.pickup[0] - Math.sin(dir) * CURB_OFFSET_M,
    y: taxi.pickup[1] + Math.cos(dir) * CURB_OFFSET_M,
  }
  // 車が来る方向（経路の上流）を向いて待つ。道路の真横を向くと、
  // 乗車地点の目印が視界の真ん中に立ってしまう
  const blocked = touchesBuilding(index, curb.x, curb.y)
  placePedestrian(
    blocked ? taxi.pickup[0] : curb.x,
    blocked ? taxi.pickup[1] : curb.y,
    dir + Math.PI,
  )
}

export function Pedestrian() {
  const palette = usePalette()
  const gl = useThree((s) => s.gl)
  const map = useSimStore((s) => s.map)
  const phase = useSimStore((s) => s.taxi.phase)

  const index = useMemo<BuildingIndex | null>(
    () => (map ? buildBuildingIndex(map.buildings) : null),
    [map],
  )

  const bodyRefs = useRef<(THREE.Mesh | null)[]>([])
  const limbRefs = useRef<(THREE.Mesh | null)[]>([])
  const rootRef = useRef<THREE.Group>(null)
  const poseSentAt = useRef(0)
  const poseOnServer = useRef(false)

  const scratch = useMemo(
    () => ({
      pose: createPose(),
      transform: createPedestrianScratch(),
      base: new THREE.Matrix4(),
      out: new THREE.Matrix4(),
      blockers: [] as VehicleBlocker[],
    }),
    [],
  )

  // ★ palette は `args` に渡さない（マテリアルごと作り直されると消える。code_review S-01）
  const resources = useMemo(() => {
    const head = makeHeadGeometry()
    const torso = makeTorsoGeometry()
    const hip = makeHipGeometry()
    const arm = makeArmGeometry()
    const leg = makeLegGeometry()
    const skin = new THREE.MeshStandardMaterial({ roughness: 0.75, metalness: 0.05 })
    const top = new THREE.MeshStandardMaterial({ roughness: 0.68, metalness: 0.08 })
    const bottom = new THREE.MeshStandardMaterial({ roughness: 0.8, metalness: 0.05 })
    return { head, torso, hip, arm, leg, skin, top, bottom }
  }, [])

  useEffect(() => {
    resources.skin.color.set(palette.pedestrianSkin)
    resources.top.color.set(palette.pedestrianTop)
    resources.bottom.color.set(palette.pedestrianBottom)
  }, [resources, palette])

  useEffect(() => {
    const r = resources
    return () => {
      r.head.dispose()
      r.torso.dispose()
      r.hip.dispose()
      r.arm.dispose()
      r.leg.dispose()
      r.skin.dispose()
      r.top.dispose()
      r.bottom.dispose()
    }
  }, [resources])

  // 実用モードを抜けるまで状態を持ち越す。
  // ★ 抜けるときは**必ず街から消すこと**。残すと開発モードに戻ったあとも、
  //   見えない人の前で車が止まり続ける（サーバー側の TTL では 1 秒かかる）
  useEffect(
    () => () => {
      send({ type: 'player_pose', at: null })
      resetPedestrian()
    },
    [],
  )

  // 配車の段階に合わせて、待つ位置・乗る・降りるを切り替える
  useEffect(() => {
    const store = useSimStore.getState()
    const taxi = store.taxi

    if (isRidingPhase(phase)) {
      pedestrian.riding = true
      pedestrian.ridingVehicle = taxi.vehicleId
      // 乗車中は既存の運転席カメラをそのまま使う（決定 11）
      store.setFollowTarget(taxi.vehicleId)
      store.setCameraMode('driver')
      return
    }
    store.setCameraMode('orbit')

    if (pedestrian.riding) {
      const slot = pedestrian.ridingVehicle
      const v = scratch.blockers.find((b) => b.id === slot)
      const seat =
        v ?? {
          id: slot,
          x: pedestrian.x,
          y: pedestrian.y,
          heading: pedestrian.heading,
        }
      const spot = alightPosition(seat)
      placePedestrian(spot.x, spot.y, seat.heading - Math.PI / 2)
      pedestrian.riding = false
      pedestrian.ridingVehicle = -1
      return
    }

    // ★ 既に立っているなら動かさない。乗車地点は自分の現在地なので、
    //   ここで立たせ直すと「呼んだ瞬間に自分が飛ぶ」ことになる。
    //   立たせるのはリロード・再マウントで街から消えているときだけ
    if (phase === 'approaching' && !pedestrian.placed) placeAtPickup(taxi, index)

    // 停まったタクシーの方へ向き直す。**照準に入らないと [Enter] が効かない**ので、
    // 待っている人が自分で振り向く手間を省く
    if (phase === 'waiting') {
      const car = scratch.blockers.find((b) => b.id === taxi.vehicleId)
      if (car) {
        pedestrian.heading = Math.atan2(car.y - pedestrian.y, car.x - pedestrian.x)
        pedestrian.pitch = 0
      }
    }
  }, [phase, index, scratch])

  // 視線（ポインタロック）と WASD
  useEffect(() => {
    const canvas = gl.domElement

    // ポインタロックは拒まれることがある（別ドキュメント越しの操作・ユーザー設定）。
    // 失敗しても WASD は window で拾っているので、黙って歩けるままにする
    const onClick = () => {
      if (pedestrian.riding || document.pointerLockElement === canvas) return
      try {
        const pending = canvas.requestPointerLock?.() as unknown
        if (pending instanceof Promise) pending.catch(() => undefined)
      } catch {
        /* 視線は回せないが歩行は続けられる */
      }
    }
    const onLockChange = () => {
      pedestrian.locked = document.pointerLockElement === canvas
      if (!pedestrian.locked) releasePedestrianKeys()
    }
    const onMove = (e: MouseEvent) => {
      if (!pedestrian.locked) return
      pedestrian.heading -= e.movementX * LOOK_SENSITIVITY
      pedestrian.pitch = Math.max(
        -PITCH_LIMIT,
        Math.min(PITCH_LIMIT, pedestrian.pitch - e.movementY * LOOK_SENSITIVITY),
      )
    }

    const setKey = (code: string, down: boolean): boolean => {
      const input = pedestrian.input
      switch (code) {
        case 'KeyW':
          input.forward = down
          return true
        case 'KeyS':
          input.back = down
          return true
        case 'KeyA':
          input.left = down
          return true
        case 'KeyD':
          input.right = down
          return true
        default:
          return false
      }
    }

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey || e.isComposing) return
      if (isTypingTarget(e.target)) return
      if (setKey(e.code, true)) {
        e.preventDefault()
        return
      }
      if (e.code === 'Enter') {
        e.preventDefault()
        handleEnter()
      } else if (e.code === 'Space') {
        e.preventDefault()
        send({ type: 'cancel_taxi', halt: true })
      }
    }
    const onKeyUp = (e: KeyboardEvent) => {
      if (setKey(e.code, false)) e.preventDefault()
    }
    const onBlur = () => releasePedestrianKeys()

    canvas.addEventListener('click', onClick)
    document.addEventListener('pointerlockchange', onLockChange)
    window.addEventListener('mousemove', onMove)
    window.addEventListener('keydown', onKeyDown)
    window.addEventListener('keyup', onKeyUp)
    window.addEventListener('blur', onBlur)
    return () => {
      canvas.removeEventListener('click', onClick)
      document.removeEventListener('pointerlockchange', onLockChange)
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('keydown', onKeyDown)
      window.removeEventListener('keyup', onKeyUp)
      window.removeEventListener('blur', onBlur)
      if (document.pointerLockElement === canvas) document.exitPointerLock?.()
      pedestrian.locked = false
      releasePedestrianKeys()
    }
  }, [gl])

  useFrame((_state, delta) => {
    const root = rootRef.current
    if (!root) return

    // サーバーは徒歩キャラを NPC 歩行者と同じ 1 人として扱う。
    // 送らないと、車は目の前に立っても止まらず、轢いてもすり抜ける
    const reportAway = () => {
      if (!poseOnServer.current) return
      if (send({ type: 'player_pose', at: null })) poseOnServer.current = false
    }
    const reportHere = () => {
      const now = performance.now()
      if (now - poseSentAt.current < POSE_REPORT_MS) return
      poseSentAt.current = now
      if (send({ type: 'player_pose', at: [pedestrian.x, pedestrian.y] })) {
        poseOnServer.current = true
      }
    }

    const blockers = collectBlockers(scratch.blockers, scratch.pose)

    // ★ 立たせるのは**走っている車が見えてから**。道路上だと保証できるのがそれだけで、
    //   原点に置くと建物の中から始まることがある（画面が真っ白になる）。
    //   配車の途中でリロードされたときは、待っていた乗車地点へ戻す
    if (!pedestrian.placed) {
      root.visible = false
      reportAway()
      const taxi = useSimStore.getState().taxi
      if (isBoardablePhase(taxi.phase)) {
        placeAtPickup(taxi, index)
        return
      }
      const first = blockers[0]
      if (first) {
        placePedestrian(
          first.x - Math.cos(first.heading) * SPAWN_BEHIND_M,
          first.y - Math.sin(first.heading) * SPAWN_BEHIND_M,
          first.heading,
        )
      }
      return
    }

    const visible = !pedestrian.riding
    root.visible = visible
    if (!visible) {
      pedestrian.speed = 0
      pedestrian.aimed = -1
      // 乗車中は車内にいる。送り続けると、乗せた車が乗客を歩行者と見て止まる
      reportAway()
      return
    }
    reportHere()

    const input = pedestrian.input
    let ax = 0
    let ay = 0
    if (input.forward) ax += 1
    if (input.back) ax -= 1
    if (input.left) ay += 1
    if (input.right) ay -= 1

    const magnitude = Math.hypot(ax, ay)
    const step = Math.min(delta, 0.1)
    if (magnitude > 0) {
      const speed = WALK_SPEED_MPS
      const cos = Math.cos(pedestrian.heading)
      const sin = Math.sin(pedestrian.heading)
      const nx = ax / magnitude
      const ny = ay / magnitude
      const dx = (cos * nx - sin * ny) * speed * step
      const dy = (sin * nx + cos * ny) * speed * step
      const moved = resolveMove(index, pedestrian.x, pedestrian.y, dx, dy, blockers)
      const actual = Math.hypot(moved.x - pedestrian.x, moved.y - pedestrian.y)
      pedestrian.x = moved.x
      pedestrian.y = moved.y
      pedestrian.speed = step > 0 ? actual / step : 0
      pedestrian.stride += actual * STRIDE_PER_METER
    } else {
      pedestrian.speed = 0
      // 止まったら手足を中立へ戻す
      pedestrian.stride += (Math.round(pedestrian.stride / Math.PI) * Math.PI - pedestrian.stride) * Math.min(1, step * 8)
    }

    pedestrian.aimed = aimedVehicle(
      pedestrian.x,
      pedestrian.y,
      pedestrian.heading,
      blockers,
    )

    const intensity = Math.min(1, pedestrian.speed / WALK_SPEED_MPS)
    const swing = limbSwing(pedestrian.stride, intensity)
    pedestrian.bob = swing.bob
    composePedestrianMatrix(
      scratch.transform,
      pedestrian.x,
      pedestrian.y,
      pedestrian.heading,
      swing.bob,
      scratch.base,
    )

    for (const mesh of bodyRefs.current) {
      if (!mesh) continue
      mesh.matrix.copy(scratch.base)
      mesh.matrixWorldNeedsUpdate = true
    }
    for (let i = 0; i < LIMB_JOINTS.length; i++) {
      const mesh = limbRefs.current[i]
      if (!mesh) continue
      composeLimbMatrix(scratch.transform, scratch.base, i, swingFor(i, swing), scratch.out)
      mesh.matrix.copy(scratch.out)
      mesh.matrixWorldNeedsUpdate = true
    }
  })

  return (
    <group ref={rootRef} visible={false}>
      <mesh
        ref={(m) => {
          bodyRefs.current[0] = m
        }}
        geometry={resources.head}
        material={resources.skin}
        matrixAutoUpdate={false}
        frustumCulled={false}
        castShadow
      />
      <mesh
        ref={(m) => {
          bodyRefs.current[1] = m
        }}
        geometry={resources.torso}
        material={resources.top}
        matrixAutoUpdate={false}
        frustumCulled={false}
        castShadow
      />
      <mesh
        ref={(m) => {
          bodyRefs.current[2] = m
        }}
        geometry={resources.hip}
        material={resources.bottom}
        matrixAutoUpdate={false}
        frustumCulled={false}
        castShadow
      />
      {LIMB_JOINTS.map((joint, i) => (
        <mesh
          key={`${joint.kind}-${joint.side}`}
          ref={(m) => {
            limbRefs.current[i] = m
          }}
          geometry={joint.kind === 'arm' ? resources.arm : resources.leg}
          material={joint.kind === 'arm' ? resources.top : resources.bottom}
          matrixAutoUpdate={false}
          frustumCulled={false}
          castShadow
        />
      ))}
    </group>
  )
}

/** [Enter]: 照準の車両に乗る／降りる。 */
function handleEnter(): void {
  const { taxi } = useSimStore.getState()
  if (pedestrian.riding) {
    send({ type: 'alight_taxi' })
    return
  }
  if (!isBoardablePhase(taxi.phase)) return
  if (pedestrian.aimed !== taxi.vehicleId) return
  send({ type: 'board_taxi' })
}
