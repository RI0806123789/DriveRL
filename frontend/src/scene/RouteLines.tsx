/** 各車両の進路（太い矢印）と目的地マーカー。 */

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
  /** 解放待ちのジオメトリ。useFrame の中で即 dispose すると、React が */
  const pendingDispose = useRef<THREE.BufferGeometry[]>([])

  const retire = (entry: RouteEntry | undefined) => {
    if (!entry) return
    if (entry.ribbon) pendingDispose.current.push(entry.ribbon)
    if (entry.chevrons) pendingDispose.current.push(entry.chevrons)
  }

  useFrame(() => {
    if (!showRoutes) return
    if (frameBuffer.routeVersion === lastRouteVersion.current) return
    lastRouteVersion.current = frameBuffer.routeVersion

    const cache = held.current
    let dirty = false

    for (const id of Array.from(cache.keys())) {
      if (!frameBuffer.routes.has(id)) {
        retire(cache.get(id)?.entry)
        cache.delete(id)
        dirty = true
      }
    }

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

  useEffect(() => {
    if (pendingDispose.current.length === 0) return
    for (const g of pendingDispose.current) g.dispose()
    pendingDispose.current = []
  })

  useEffect(() => {
    if (showRoutes) {
      lastRouteVersion.current = -1
      return
    }
    for (const h of held.current.values()) retire(h.entry)
    held.current.clear()
    setRoutes([])
  }, [showRoutes])

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
