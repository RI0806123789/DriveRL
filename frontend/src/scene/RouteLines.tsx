/**
 * 各車両の進路（太い矢印）と目的地マーカー。
 *
 * カーナビの案内表示のように、経路を**幅のあるリボン**として描き、
 * 一定間隔で進行方向を示す矢羽根（シェブロン）を重ねる。
 * 運転席視点にしたときに「次にどちらへ行くのか」が一目で分かることを狙っている。
 *
 * 経路（frameBuffer.routes）はバックエンドが「変化したフレームだけ」送る。
 * ★ 作り直しは**変わった車両だけ**に限る。routeVersion は「どれか 1 台でも変わった」
 *   としか言わないので、これで全台を作り直すと 64 台 × 75〜250 点を毎秒 4〜5 回
 *   作っては捨てることになり、描画が周期的に引っかかる（code_review F-01）。
 *   車両ごとの版は frameBuffer.routeRevisions が持つ。
 * ★ 経路線を OFF にしている間は**そもそも作らない**。以前は描画の可否しか見ておらず、
 *   OFF でも生成コストが丸ごと残っていた。
 *
 * 目的地マーカーは毎フレーム動きうるが React を回したくないので ref を直接書き換える。
 */

import { memo, useEffect, useMemo, useRef, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { frameBuffer } from '../store/frameBuffer'
import { buildChevrons, buildRibbon } from './routeArrowGeometry'
import { vehicleColor } from './vehicleColors'

interface RouteEntry {
  id: number
  ribbon: THREE.BufferGeometry | null
  chevrons: THREE.BufferGeometry | null
}

/** キャッシュ 1 件。rev は「このジオメトリを作った時点の車両ごとの版」 */
interface HeldEntry {
  entry: RouteEntry
  rev: number
}

// ---------------------------------------------------------------------------

export interface RouteLinesProps {
  showRoutes: boolean
  showGoals: boolean
  maxVehicles: number
}

export function RouteLines({ showRoutes, showGoals, maxVehicles }: RouteLinesProps) {
  const [routes, setRoutes] = useState<RouteEntry[]>([])
  const lastRouteVersion = useRef(-1)
  /** id -> 生成済みジオメトリ。React の外で持ち、差分だけ差し替える */
  const held = useRef<Map<number, HeldEntry>>(new Map())
  /**
   * 解放待ちのジオメトリ。useFrame の中で即 dispose すると、React が
   * 差し替えを反映する前に 1 回描画されて GPU バッファを無駄に作り直す。
   * コミット後の useEffect でまとめて解放する。
   */
  const pendingDispose = useRef<THREE.BufferGeometry[]>([])

  const retire = (entry: RouteEntry | undefined) => {
    if (!entry) return
    if (entry.ribbon) pendingDispose.current.push(entry.ribbon)
    if (entry.chevrons) pendingDispose.current.push(entry.chevrons)
  }

  useFrame(() => {
    // OFF の間は作らない（描画しないものに生成コストを払わない）
    if (!showRoutes) return
    if (frameBuffer.routeVersion === lastRouteVersion.current) return
    lastRouteVersion.current = frameBuffer.routeVersion

    const cache = held.current
    let dirty = false

    // 1. 消えた経路（非アクティブ化・マップ切替）を手放す
    for (const id of Array.from(cache.keys())) {
      if (!frameBuffer.routes.has(id)) {
        retire(cache.get(id)?.entry)
        cache.delete(id)
        dirty = true
      }
    }

    // 2. 版が上がった経路だけ作り直す
    frameBuffer.routes.forEach((points, id) => {
      const rev = frameBuffer.routeRevisions.get(id) ?? 0
      const current = cache.get(id)
      if (current && current.rev === rev) return
      retire(current?.entry)
      cache.set(id, {
        rev,
        entry: { id, ribbon: buildRibbon(points), chevrons: buildChevrons(points) },
      })
      dirty = true
    })

    if (dirty) setRoutes(Array.from(cache.values(), (h) => h.entry))
  })

  // 差し替えが React に反映された後で、置き換えられたジオメトリを解放する
  useEffect(() => {
    if (pendingDispose.current.length === 0) return
    for (const g of pendingDispose.current) g.dispose()
    pendingDispose.current = []
  })

  // 経路線を OFF にしたら GPU 上のジオメトリを手放す。
  // ON に戻したときは版を無効化して全件を作り直す（OFF 中の変化を取りこぼさない）
  useEffect(() => {
    if (showRoutes) {
      lastRouteVersion.current = -1
      return
    }
    for (const h of held.current.values()) retire(h.entry)
    held.current.clear()
    setRoutes([])
  }, [showRoutes])

  // アンマウント時の取りこぼしを防ぐ
  useEffect(() => {
    const cache = held.current
    const pending = pendingDispose.current
    return () => {
      for (const h of cache.values()) {
        h.entry.ribbon?.dispose()
        h.entry.chevrons?.dispose()
      }
      cache.clear()
      for (const g of pending) g.dispose()
      pending.length = 0
    }
  }, [])

  return (
    <group>
      {showRoutes && routes.map((r) => <RouteArrow key={`route-${r.id}`} entry={r} />)}
      {showGoals && <GoalMarkers maxVehicles={maxVehicles} />}
    </group>
  )
}

/** 1 台分の進路矢印 */
const RouteArrow = memo(function RouteArrow({ entry }: { entry: RouteEntry }) {
  const color = useMemo(() => new THREE.Color(vehicleColor(entry.id)), [entry.id])

  // ★ polygonOffset で路面・標示より確実に手前へ出す。
  //
  //   進路ラインは路面から 9cm しか浮いていない（y=0.11 対 0.02）。
  //   カメラは near=0.5 / far=8000 なので、24bit の深度バッファでも
  //   **約 615m より遠いと 9cm が路面の polygonOffset(-1/-2) を下回り**、
  //   深度テストで捨てられて線が丸ごと消える（縮小すると見えなくなる）。
  //   高さを上げる案は、運転席視点で線が宙に浮いて見えるので採らない。
  //
  //   depthTest は切らないこと。切ると建物の裏を通る経路まで透けて見える
  //   （追跡ピンだけは「見失うほうが困る」ので意図的に切ってある）。
  //
  //   順序: 地面 < 路面 -1/-2 < 中央線 -2/-4 < 標示 -3/-6 < リボン < 矢羽根
  const ribbonMaterial = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        color: color.clone(),
        transparent: true,
        opacity: 0.28,
        depthWrite: false,
        side: THREE.DoubleSide,
        toneMapped: false,
        polygonOffset: true,
        polygonOffsetFactor: -4,
        polygonOffsetUnits: -8,
      }),
    [color],
  )
  const chevronMaterial = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        color: color.clone(),
        transparent: true,
        opacity: 0.85,
        depthWrite: false,
        side: THREE.DoubleSide,
        toneMapped: false,
        polygonOffset: true,
        polygonOffsetFactor: -5,
        polygonOffsetUnits: -10,
      }),
    [color],
  )

  useEffect(() => {
    return () => {
      ribbonMaterial.dispose()
      chevronMaterial.dispose()
    }
  }, [ribbonMaterial, chevronMaterial])

  return (
    <group>
      {entry.ribbon && (
        <mesh geometry={entry.ribbon} material={ribbonMaterial} renderOrder={2} />
      )}
      {entry.chevrons && (
        <mesh geometry={entry.chevrons} material={chevronMaterial} renderOrder={3} />
      )}
    </group>
  )
})

// ---------------------------------------------------------------------------
// 目的地マーカー
// ---------------------------------------------------------------------------

const ringGeom = new THREE.TorusGeometry(3.2, 0.22, 8, 40)
ringGeom.rotateX(-Math.PI / 2)
const beamGeom = new THREE.CylinderGeometry(0.22, 0.22, 1, 10)
beamGeom.translate(0, 0.5, 0)

const BEAM_HEIGHT = 14

const GoalMarker = memo(function GoalMarker({ id }: { id: number }) {
  const group = useRef<THREE.Group>(null)
  const color = useMemo(() => new THREE.Color(vehicleColor(id)), [id])

  const material = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        color: color.clone(),
        transparent: true,
        opacity: 0.5,
        // リボンと同じ理由。路面に置くものは手前へ押さないと遠景で消える
        polygonOffset: true,
        polygonOffsetFactor: -4,
        polygonOffsetUnits: -8,
      }),
    [color],
  )
  useEffect(() => () => material.dispose(), [material])

  useFrame(() => {
    const g = group.current
    if (!g) return
    const curr = frameBuffer.curr
    if (!curr) {
      g.visible = false
      return
    }
    const v =
      curr.vehicles[id]?.id === id ? curr.vehicles[id] : curr.vehicles.find((x) => x.id === id)
    if (!v || !v.active) {
      g.visible = false
      return
    }
    g.visible = true
    g.position.set(v.goal[0], 0, -v.goal[1])
    // ゆっくり回して目に留まりやすくする
    g.rotation.y = performance.now() * 0.0008
  })

  return (
    <group ref={group} visible={false}>
      <mesh geometry={ringGeom} material={material} position={[0, 0.12, 0]} />
      <mesh geometry={beamGeom} material={material} scale={[1, BEAM_HEIGHT, 1]} />
    </group>
  )
})

const GoalMarkers = memo(function GoalMarkers({ maxVehicles }: { maxVehicles: number }) {
  const slots = useMemo(
    () => Array.from({ length: Math.max(0, maxVehicles) }, (_, i) => i),
    [maxVehicles],
  )
  return (
    <group>
      {slots.map((id) => (
        <GoalMarker key={`goal-${id}`} id={id} />
      ))}
    </group>
  )
})
