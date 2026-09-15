/** 左側のシミュレーター画面（memo 4章）。 */

import { useEffect, useRef } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { Buildings } from './Buildings'
import { CameraRig } from './CameraRig'
import { DetectionOverlay } from './DetectionOverlay'
import { Ground } from './Ground'
import { InteractionPlane } from './InteractionPlane'
import { LaneDetectionOverlay } from './LaneDetectionOverlay'
import { Obstacles } from './Obstacles'
import { RoadMarkings } from './RoadMarkings'
import { RoadNetwork } from './RoadNetwork'
import { RouteLines } from './RouteLines'
import { SpeedSigns } from './SpeedSigns'
import { TrafficSignals } from './TrafficSignals'
import { Vehicles } from './Vehicles'
import { DARK_SCENE } from './palette'
import { usePalette } from './usePalette'
import { sceneStats } from './sceneStats'
import { useSimStore } from '../store/simStore'
import { ConeIcon, CarIcon, MapIcon } from '../ui/Icons'
import type { MapBounds } from '../types/protocol'

/** 影を落とす範囲をマップの大きさに合わせる */
function shadowExtent(bounds: MapBounds | null): number {
  if (!bounds) return 400
  return Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY) * 0.62 + 80
}

/** フォグを調整する基準になるマップの一辺 [m]。 */
const DESIGN_SPAN_M = 900

/** マップの一辺（大きいほう）[m]。無ければ 400m プリセット相当 */
function mapSpan(bounds: MapBounds | null): number {
  if (!bounds) return DESIGN_SPAN_M
  return Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY)
}

/** カメラの far 面 [m]。 */
function cameraFarFor(span: number): number {
  return Math.max(8000, span * 2.5)
}

export function SimulatorView() {
  const map = useSimStore((s) => s.map)
  const view = useSimStore((s) => s.view)
  const maxVehicles = useSimStore((s) => s.config.maxVehicles)
  const interaction = useSimStore((s) => s.interaction)
  const status = useSimStore((s) => s.status)
  const connection = useSimStore((s) => s.connection)

  const bounds = map?.bounds ?? null
  const extent = shadowExtent(bounds)
  const span = mapSpan(bounds)
  const cx = bounds ? (bounds.minX + bounds.maxX) / 2 : 0
  const cz = bounds ? -((bounds.minY + bounds.maxY) / 2) : 0

  return (
    <>
      <div className="stage-canvas">
        <Canvas
          shadows="percentage"
          dpr={[1, 2]}
          camera={{ fov: 50, near: 0.5, far: 8000, position: [300, 320, 300] }}
          gl={{ antialias: true, powerPreference: 'high-performance' }}
          onCreated={({ gl, scene }) => {
            gl.setClearColor(new THREE.Color(DARK_SCENE.sky))
            scene.fog = new THREE.Fog(DARK_SCENE.sky, DARK_SCENE.fogNear, DARK_SCENE.fogFar)
          }}
        >
          <SceneAtmosphere span={span} />
          <SceneLights />
          <SunLight cx={cx} cz={cz} extent={extent} castShadow={view.shadows} />

          <Ground bounds={bounds} showGrid={view.grid} receiveShadow={view.shadows} />

          {map && view.roads && <RoadNetwork edges={map.edges} receiveShadow={view.shadows} />}
          {map && view.markings && (
            <RoadMarkings edges={map.edges} signals={map.signals ?? []} />
          )}
          {map && view.signals && (
            <TrafficSignals
              signals={map.signals ?? []}
              nodes={map.nodes}
              castShadow={view.shadows}
            />
          )}
          <SpeedSigns />
          {map && view.buildings && (
            <Buildings
              buildings={map.buildings}
              castShadow={view.shadows}
              receiveShadow={view.shadows}
            />
          )}

          <RouteLines
            showRoutes={view.routes}
            showGoals={view.goals}
            maxVehicles={maxVehicles}
          />
          <LaneDetectionOverlay />
          <Vehicles maxVehicles={maxVehicles} castShadow={view.shadows} />
          <Obstacles castShadow={view.shadows} />

          <InteractionPlane bounds={bounds} />
          <CameraRig bounds={bounds} />
          <RenderStatsProbe />
        </Canvas>
      </div>

      {interaction !== 'none' && (
        <div className="stage-hint">
          {interaction === 'obstacle' ? <ConeIcon size={16} /> : <CarIcon size={16} />}
          {interaction === 'obstacle'
            ? '3D 画面をクリックすると障害物を置きます（置いた障害物をクリックすると 1 個だけ消せます / ドラッグは視点操作）'
            : '3D 画面をクリックすると車両を追加します（最寄りの道路にスナップ）'}
        </div>
      )}

      {!map && (
        <div className="stage-placeholder">
          <div className="stage-placeholder-card">
            <MapIcon size={32} />
            <div className="stage-placeholder-title">
              {connection !== 'open'
                ? 'バックエンドに接続していません'
                : status.state === 'loading_map'
                  ? '地図データを取得しています'
                  : 'マップを選択してください'}
            </div>
            <div className="stage-placeholder-body">
              {connection !== 'open' ? (
                <>
                  <code className="m3-mono">backend/run.py</code> を起動してください。
                  <br />
                  接続できるようになると自動で再接続します。
                </>
              ) : status.state === 'loading_map' ? (
                <>
                  OpenStreetMap（Overpass API）から道路網と建物を取得しています。
                  <br />
                  初回は 10〜60 秒かかります。次回以降はキャッシュから即座に読み込まれます。
                </>
              ) : (
                <>
                  右側のパネルの「マップ」タブから、走行させる都市を選んでください。
                  <br />
                  読み込みが完了すると、その場で強化学習が始まります。
                </>
              )}
            </div>
            {status.state === 'loading_map' && <div className="m3-progress" />}
          </div>
        </div>
      )}

      <DetectionOverlay />
    </>
  )
}

interface SunLightProps {
  cx: number
  cz: number
  extent: number
  castShadow: boolean
}

/** 背景色とフォグを配色に追従させる。 */
function SceneAtmosphere({ span }: { span: number }) {
  const palette = usePalette()
  const gl = useThree((s) => s.gl)
  const scene = useThree((s) => s.scene)
  const camera = useThree((s) => s.camera)

  const k = Math.max(1, span / DESIGN_SPAN_M)

  useEffect(() => {
    gl.setClearColor(new THREE.Color(palette.sky))
    const near = palette.fogNear * k
    const far = palette.fogFar * k
    if (scene.fog instanceof THREE.Fog) {
      scene.fog.color.set(palette.sky)
      scene.fog.near = near
      scene.fog.far = far
    } else {
      scene.fog = new THREE.Fog(palette.sky, near, far)
    }
  }, [gl, scene, palette, k])

  useEffect(() => {
    const far = cameraFarFor(span)
    if (camera instanceof THREE.PerspectiveCamera && camera.far !== far) {
      camera.far = far
      camera.updateProjectionMatrix()
    }
  }, [camera, span])

  return null
}

/** 環境光。`args` ではなく個別の props で渡す。 */
function SceneLights() {
  const palette = usePalette()
  return (
    <>
      <hemisphereLight
        color={palette.hemiSky}
        groundColor={palette.hemiGround}
        intensity={palette.hemiIntensity}
      />
      <ambientLight intensity={palette.ambientIntensity} />
    </>
  )
}

/** three の DirectionalLight は **target がシーングラフに入っていないと */
function SunLight({ cx, cz, extent, castShadow }: SunLightProps) {
  const palette = usePalette()
  const light = useRef<THREE.DirectionalLight>(null)
  const target = useRef<THREE.Object3D>(null)

  useEffect(() => {
    if (light.current && target.current) light.current.target = target.current
  }, [])

  return (
    <>
      <directionalLight
        ref={light}
        position={[cx + extent * 0.6, extent * 1.1, cz + extent * 0.45]}
        intensity={palette.sunIntensity}
        color={palette.sun}
        castShadow={castShadow}
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
        shadow-camera-near={1}
        shadow-camera-far={extent * 3.2}
        shadow-camera-left={-extent}
        shadow-camera-right={extent}
        shadow-camera-top={extent}
        shadow-camera-bottom={-extent}
        shadow-bias={-0.0008}
        shadow-normalBias={0.6}
      />
      <object3D ref={target} position={[cx, 0, cz]} />
    </>
  )
}

/** gl.info.render.calls を毎フレーム拾って sceneStats へ書く。 */
function RenderStatsProbe() {
  const gl = useThree((s) => s.gl)

  useEffect(() => {
    gl.info.autoReset = false
    return () => {
      gl.info.autoReset = true
    }
  }, [gl])

  useFrame(({ gl: renderer }) => {
    sceneStats.drawCalls = renderer.info.render.calls
    sceneStats.triangles = renderer.info.render.triangles
    renderer.info.reset()
  })
  return null
}
