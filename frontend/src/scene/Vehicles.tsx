/**
 * 車両。
 *
 * ・車体（ボックス）＋ノーズ＋キャビン＋4輪の簡易モデル。**前方は +X 軸**。
 *   protocol.md 1.3 の通り、これで rotation.y = heading がそのまま正しくなる。
 * ・前輪は steer の分だけ Y 軸回転させ、4 輪とも speed に応じて転がす。
 * ・active === false のスロットは行列をゼロにして消す。
 * ・collided === true の車両は赤く点滅させる（衝突判定はバックエンド側の結果）。
 * ・追従カメラの対象車両は足元のリングと頭上のピンでハイライトする。
 *
 * ★ 全車両を InstancedMesh 1 組にまとめてある（code_review F-02）。
 *   1 台 7 メッシュの通常構成だと 64 台で本体 448 + 影 384 のドローコールになり、
 *   Intel Arc 内蔵 GPU では 60fps を割る。まとめると台数によらず
 *   本体 4 回（車体・ノーズ・キャビン・車輪）＋影 3 回で済む。
 *   ハイライトのリングとピンは**同時にひとつしか出ない**ので、
 *   スロットごとに持たず 1 個だけ用意して対象車両へ移動させる。
 *
 * 位置・姿勢は scene/interpolation.ts の補間結果を毎フレーム読む。
 * zustand は経由しない（20Hz で React を再レンダリングしないため）。
 */

import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

import { useSimStore } from '../store/simStore'
import { computeAlpha, createPose, sampleVehicle } from './interpolation'
import {
} from './palette'
import { usePalette } from './usePalette'
import { vehicleColor } from './vehicleColors'
import {
  WHEEL_OFFSETS,
  WHEEL_RADIUS,
  composeVehicleMatrix,
  composeWheelMatrix,
  createTransformScratch,
  makeBodyGeometry,
  makeCabinGeometry,
  makeNoseGeometry,
  makeWheelGeometry,
} from './vehicleGeometry'

const ringGeom = new THREE.TorusGeometry(2.9, 0.08, 8, 44)
ringGeom.rotateX(-Math.PI / 2)

// --- 追跡ピン（地図のピン型：下向きの円錐＋球） ---
/** ピンの先端が指す高さ [m]。車の屋根（約 1.6m）より上 */
const PIN_TIP_Y = 2.6
const PIN_CONE_H = 1.1
const PIN_BALL_R = 0.46

const pinConeGeom = new THREE.ConeGeometry(PIN_BALL_R * 0.95, PIN_CONE_H, 16)
// ConeGeometry は +Y に尖る。先端を下へ向け、先端が原点に来るよう持ち上げる
pinConeGeom.rotateX(Math.PI)
pinConeGeom.translate(0, PIN_CONE_H / 2, 0)

const pinBallGeom = new THREE.SphereGeometry(PIN_BALL_R, 20, 14)
pinBallGeom.translate(0, PIN_CONE_H + PIN_BALL_R * 0.55, 0)


/** 車両の見た目の状態。色と自己発光を決める */
const STATE_NORMAL = 0
const STATE_GOAL = 1
const STATE_COLLIDED = 2

/**
 * MeshStandardMaterial に「インスタンスごとの自己発光」を足す。
 *
 * three は instanceColor（＝ color への乗算）しか用意しておらず、emissive は
 * マテリアル全体で 1 つしか持てない。衝突時の赤い明滅は暗いシーンでは
 * 自己発光あってこそ見えるので、属性を 1 本足して emissive に加算する。
 * 置換対象の文字列は three 0.185 の meshphysical.glsl.js に存在することを確認済み。
 */
function attachInstanceEmissive(material: THREE.MeshStandardMaterial): void {
  material.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader
      .replace(
        '#include <common>',
        '#include <common>\nattribute vec3 instanceEmissive;\nvarying vec3 vInstanceEmissive;',
      )
      .replace(
        '#include <begin_vertex>',
        '#include <begin_vertex>\nvInstanceEmissive = instanceEmissive;',
      )
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vInstanceEmissive;')
      .replace(
        'vec3 totalEmissiveRadiance = emissive;',
        'vec3 totalEmissiveRadiance = emissive + vInstanceEmissive;',
      )
  }
  // シェーダーを差し替えたことをプログラムキャッシュのキーに反映させる
  material.customProgramCacheKey = () => 'instanceEmissive'
}

export interface VehiclesProps {
  maxVehicles: number
  castShadow: boolean
}

export function Vehicles({ maxVehicles, castShadow }: VehiclesProps) {
  const palette = usePalette()

  // 衝突中・到達時に寄せる色。毎フレーム new しないよう配色ごとに 1 度だけ作る
  const stateColors = useMemo(
    () => ({
      collided: new THREE.Color(palette.vehicleCollided),
      /** 目的地に着いたときの自己発光。palette の色を薄めて使う */
      reachedEmissive: new THREE.Color(palette.vehicleReached).multiplyScalar(0.35),
    }),
    [palette.vehicleCollided, palette.vehicleReached],
  )

  // 擬似固定エージェント数方式（memo 5章）: スロット数ぶんだけ常に用意し、
  // active === false のときは行列をゼロにして消す。
  const count = Math.max(1, maxVehicles)

  const bodyRef = useRef<THREE.InstancedMesh>(null)
  const noseRef = useRef<THREE.InstancedMesh>(null)
  const cabinRef = useRef<THREE.InstancedMesh>(null)
  const wheelRef = useRef<THREE.InstancedMesh>(null)
  const ringRef = useRef<THREE.Mesh>(null)
  const pinRef = useRef<THREE.Group>(null)

  const pose = useMemo(createPose, [])
  const spin = useRef(0)
  /** 直前に色を書いたときの状態。変わったときだけ GPU へ送る */
  const lastState = useRef<Int8Array>(new Int8Array(0))

  // ---- 台数が変わったら作り直す（InstancedMesh の count は固定のため） ----
  const resources = useMemo(() => {
    const bodyGeometry = makeBodyGeometry()
    const noseGeometry = makeNoseGeometry()
    const cabinGeometry = makeCabinGeometry()
    const wheelGeometry = makeWheelGeometry()

    // 車体とノーズは同じ色・同じ自己発光を使うが、three のインスタンス属性は
    // ジオメトリに載るので、それぞれに 1 本ずつ用意して同じ値を書く
    const bodyEmissive = new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3)
    const noseEmissive = new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3)
    bodyGeometry.setAttribute('instanceEmissive', bodyEmissive)
    noseGeometry.setAttribute('instanceEmissive', noseEmissive)

    const bodyMaterial = new THREE.MeshStandardMaterial({
      color: '#ffffff', // 実際の色は instanceColor で与える
      roughness: 0.4,
      metalness: 0.45,
    })
    attachInstanceEmissive(bodyMaterial)

    // ★ 色は下の useEffect が配色に合わせて書き換える。
    //    この useMemo は count だけに依存させる（配色を deps に入れると
    //    日の出・日の入りのたびに InstancedMesh ごと作り直すことになる）。
    const glassMaterial = new THREE.MeshStandardMaterial({
      color: palette.vehicleGlass,
      roughness: 0.25,
      metalness: 0.6,
    })
    const wheelMaterial = new THREE.MeshStandardMaterial({
      color: palette.vehicleWheel,
      roughness: 0.9,
      metalness: 0.1,
    })
    const ringMaterial = new THREE.MeshBasicMaterial({
      color: palette.vehicleHighlight,
      transparent: true,
      opacity: 0.85,
    })
    // 追跡ピンは建物の陰に入っても見えてほしいので深度テストを切って最後に描く。
    // 「どこにいるか」を追うための表示なので、遮蔽されて見失うほうが困る。
    // 不透明のままだと three の描画順で進路リボン（半透明）に塗りつぶされるため、
    // 透明扱いにして renderOrder を最大にする。
    const pinMaterial = new THREE.MeshBasicMaterial({
      color: '#ffffff',
      toneMapped: false,
      depthTest: false,
      depthWrite: false,
      transparent: true,
      opacity: 1,
    })

    /** 車体色。スロット番号ごとに固定（パネル側の一覧と揃うこと） */
    const baseColors = Array.from({ length: count }, (_, i) => new THREE.Color(vehicleColor(i)))

    return {
      bodyGeometry,
      noseGeometry,
      cabinGeometry,
      wheelGeometry,
      bodyEmissive,
      noseEmissive,
      bodyMaterial,
      glassMaterial,
      wheelMaterial,
      ringMaterial,
      pinMaterial,
      baseColors,
    }
  }, [count])

  // 台数が変わったら「前回書いた状態」も作り直す（-1 は未記入の意味）
  useEffect(() => {
    lastState.current = new Int8Array(count).fill(-1)
  }, [count, resources])

  // 配色が変わったらマテリアルの色だけ書き換える。
  // InstancedMesh ごと作り直すと 64 台ぶんの行列を張り直すことになるので触らない。
  // 車体色は「状態が変わったときだけ」書いているため、前回値の記録も捨てて
  // 次のフレームで必ず塗り直させる（これが無いと衝突色だけ古いまま残る）。
  useEffect(() => {
    resources.glassMaterial.color.set(palette.vehicleGlass)
    resources.wheelMaterial.color.set(palette.vehicleWheel)
    resources.ringMaterial.color.set(palette.vehicleHighlight)
    lastState.current.fill(-1)
  }, [resources, palette])

  useEffect(() => {
    const r = resources
    return () => {
      r.bodyGeometry.dispose()
      r.noseGeometry.dispose()
      r.cabinGeometry.dispose()
      r.wheelGeometry.dispose()
      r.bodyMaterial.dispose()
      r.glassMaterial.dispose()
      r.wheelMaterial.dispose()
      r.ringMaterial.dispose()
      r.pinMaterial.dispose()
    }
  }, [resources])

  // 使い回す作業用オブジェクト（毎フレームの new を避ける）
  const scratch = useMemo(
    () => ({
      transform: createTransformScratch(),
      base: new THREE.Matrix4(),
      out: new THREE.Matrix4(),
      hidden: new THREE.Matrix4().makeScale(0, 0, 0),
      color: new THREE.Color(),
      emissive: new THREE.Color(),
    }),
    [],
  )

  useFrame((_state, delta) => {
    const body = bodyRef.current
    const nose = noseRef.current
    const cabin = cabinRef.current
    const wheel = wheelRef.current
    if (!body || !nose || !cabin || !wheel) return
    if (lastState.current.length !== count) return

    const store = useSimStore.getState()
    const paused = store.status.renderPaused
    const alpha = computeAlpha(performance.now(), paused)
    const followTarget = store.followTarget
    const driverView = store.cameraMode === 'driver'

    const now = performance.now()
    const flash = 0.5 + 0.5 * Math.sin(now * 0.018)
    let colorDirty = false
    let highlightX = 0
    let highlightY = 0
    let hasHighlight = false
    // 車輪の転がりは代表車（追従対象、無ければ先頭の走行中）の速度で回す
    let rollSpeed = 0

    for (let id = 0; id < count; id++) {
      const ok = sampleVehicle(id, alpha, pose)
      // 運転席視点のとき、自分の車体はカメラの内側にあって視界を塞ぐので隠す
      const hiddenAsEgo = driverView && followTarget === id
      if (!ok || hiddenAsEgo) {
        body.setMatrixAt(id, scratch.hidden)
        nose.setMatrixAt(id, scratch.hidden)
        cabin.setMatrixAt(id, scratch.hidden)
        for (let k = 0; k < 4; k++) wheel.setMatrixAt(id * 4 + k, scratch.hidden)
        continue
      }

      if (rollSpeed === 0 || id === followTarget) rollSpeed = pose.speed

      // ENU -> Three（protocol.md 1.3）。行列の組み立ては vehicleGeometry.ts
      composeVehicleMatrix(scratch.transform, pose.x, pose.y, pose.heading, scratch.base)

      // 車体・ノーズ・キャビンはオフセットをジオメトリに焼き込んであるので同じ行列
      body.setMatrixAt(id, scratch.base)
      nose.setMatrixAt(id, scratch.base)
      cabin.setMatrixAt(id, scratch.base)

      // 車輪。前輪だけ舵角を効かせ、4 輪とも転がす
      for (let k = 0; k < WHEEL_OFFSETS.length; k++) {
        composeWheelMatrix(
          scratch.transform,
          scratch.base,
          k,
          pose.steer,
          spin.current,
          scratch.out,
        )
        wheel.setMatrixAt(id * 4 + k, scratch.out)
      }

      // 色と自己発光。衝突中は毎フレーム明滅させるので常に書き、
      // それ以外は状態が変わったときだけ書く
      const state = pose.collided ? STATE_COLLIDED : pose.reachedGoal ? STATE_GOAL : STATE_NORMAL
      if (state === STATE_COLLIDED || lastState.current[id] !== state) {
        lastState.current[id] = state
        colorDirty = true
        const base = resources.baseColors[id]
        if (state === STATE_COLLIDED) {
          scratch.color.copy(base).lerp(stateColors.collided, 0.75)
          scratch.emissive.copy(stateColors.collided).multiplyScalar(0.35 + 0.45 * flash)
        } else if (state === STATE_GOAL) {
          scratch.color.copy(base)
          scratch.emissive.copy(stateColors.reachedEmissive)
        } else {
          scratch.color.copy(base)
          scratch.emissive.setRGB(0, 0, 0)
        }
        body.setColorAt(id, scratch.color)
        nose.setColorAt(id, scratch.color)
        const e = id * 3
        resources.bodyEmissive.array[e] = scratch.emissive.r
        resources.bodyEmissive.array[e + 1] = scratch.emissive.g
        resources.bodyEmissive.array[e + 2] = scratch.emissive.b
        resources.noseEmissive.array[e] = scratch.emissive.r
        resources.noseEmissive.array[e + 1] = scratch.emissive.g
        resources.noseEmissive.array[e + 2] = scratch.emissive.b
      }

      if (id === followTarget) {
        hasHighlight = true
        highlightX = pose.x
        highlightY = pose.y
      }
    }

    // 車輪の転がり（一時停止中は進めない）
    if (!paused) spin.current -= (rollSpeed * delta) / WHEEL_RADIUS

    body.instanceMatrix.needsUpdate = true
    nose.instanceMatrix.needsUpdate = true
    cabin.instanceMatrix.needsUpdate = true
    wheel.instanceMatrix.needsUpdate = true
    if (colorDirty) {
      if (body.instanceColor) body.instanceColor.needsUpdate = true
      if (nose.instanceColor) nose.instanceColor.needsUpdate = true
      resources.bodyEmissive.needsUpdate = true
      resources.noseEmissive.needsUpdate = true
    }

    // 追従対象のハイライト。同時にひとつしか出ないので 1 個だけ動かす
    const ring = ringRef.current
    if (ring) {
      ring.visible = hasHighlight
      if (hasHighlight) {
        ring.position.set(highlightX, 0.06, -highlightY)
        ring.scale.setScalar(1 + 0.05 * Math.sin(now * 0.004))
      }
    }
    const pin = pinRef.current
    if (pin) {
      pin.visible = hasHighlight
      if (hasHighlight) {
        // ゆっくり上下させて見つけやすくする
        const bob = 0.18 * Math.sin(now * 0.0026)
        pin.position.set(highlightX, PIN_TIP_Y + bob, -highlightY)
        const target = resources.baseColors[followTarget]
        if (target) resources.pinMaterial.color.copy(target)
      }
    }
  })

  return (
    <group>
      {/* 車体（前方 +X）。色は instanceColor、明滅は instanceEmissive */}
      <instancedMesh
        key={`body-${count}`}
        ref={bodyRef}
        args={[resources.bodyGeometry, resources.bodyMaterial, count]}
        castShadow={castShadow}
        frustumCulled={false}
      />
      {/* ノーズ。元の実装どおり影は落とさない */}
      <instancedMesh
        key={`nose-${count}`}
        ref={noseRef}
        args={[resources.noseGeometry, resources.bodyMaterial, count]}
        frustumCulled={false}
      />
      {/* キャビン */}
      <instancedMesh
        key={`cabin-${count}`}
        ref={cabinRef}
        args={[resources.cabinGeometry, resources.glassMaterial, count]}
        castShadow={castShadow}
        frustumCulled={false}
      />
      {/* 車輪 4 本 × 台数 */}
      <instancedMesh
        key={`wheel-${count}`}
        ref={wheelRef}
        args={[resources.wheelGeometry, resources.wheelMaterial, count * 4]}
        castShadow={castShadow}
        frustumCulled={false}
      />

      {/* 追従対象のハイライトリング（1 個だけ用意して動かす） */}
      <mesh ref={ringRef} geometry={ringGeom} material={resources.ringMaterial} visible={false} />

      {/* 追跡ピン（パネルで選んだ車両の上に立つ） */}
      <group ref={pinRef} visible={false}>
        <mesh geometry={pinConeGeom} material={resources.pinMaterial} renderOrder={999} />
        <mesh geometry={pinBallGeom} material={resources.pinMaterial} renderOrder={999} />
      </group>
    </group>
  )
}
