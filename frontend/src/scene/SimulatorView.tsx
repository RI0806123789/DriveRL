/**
 * 左側のシミュレーター画面（memo 4章）。
 *
 * Canvas と、その上に重ねる案内表示（マップ未読込のプレースホルダ、介入モードの
 * ヒント、走行状況の HUD）をまとめて持つ。
 *
 * 3D の中身は zustand を極力読まない。20Hz の frame は frameBuffer から
 * useFrame で直接読むので、ここで再レンダリングされるのは
 * 「マップが変わった」「表示トグルを操作した」ときだけ。
 * カメラのモードと追従対象は CameraRig が自分で購読する（ここで購読すると、
 * 車両一覧をクリックするたびにシーン全体の差分計算が走ってしまうため）。
 */

import { useEffect, useRef, useState } from 'react'
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
import { currentCameraNotice, sceneStats } from './sceneStats'
import { frameBuffer } from '../store/frameBuffer'
import { useSimStore } from '../store/simStore'
import { ConeIcon, CarIcon, MapIcon } from '../ui/Icons'
import type { MapBounds } from '../types/protocol'

/** 影を落とす範囲をマップの大きさに合わせる */
function shadowExtent(bounds: MapBounds | null): number {
  if (!bounds) return 400
  return Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY) * 0.62 + 80
}

/**
 * フォグを調整する基準になるマップの一辺 [m]。
 * 400m プリセット（銀座・梅田・栄）の実測がおよそ 870〜990m なので、その付近。
 */
const DESIGN_SPAN_M = 900

/** マップの一辺（大きいほう）[m]。無ければ 400m プリセット相当 */
function mapSpan(bounds: MapBounds | null): number {
  if (!bounds) return DESIGN_SPAN_M
  return Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY)
}

/**
 * カメラの far 面 [m]。
 *
 * ★ **固定値にしないこと。** `CameraRig.overviewFor()` は俯瞰位置を
 *   `span * 0.85` を基準に取るので、俯瞰から対角の隅までは **約 1.5 * span** ある。
 *   金沢（広域・一辺 12.3km）では約 11km 離れるので、固定の 8000 では
 *   **マップ全体がクリップされて何も映らない**。
 *
 * far を伸ばしても深度精度はほとんど落ちない。分解能は
 * `z^2 * (1/near - 1/far)` に比例し、far >> near では実質 `1/near` で決まるため
 * （near=0.5 のとき far 8000 で 1.99988、far 31000 でも 1.99997 とほぼ同じ）。
 * したがって路面の polygonOffset の重ね順には影響しない。
 */
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
          // ★ shadows は "percentage"（PCFShadowMap）で固定する。
          //
          // R3F の既定（shadows={true}）は PCFSoftShadowMap だが、three 0.185 では
          // これが非推奨になっており、WebGLShadowMap.render() が毎回
          //   「PCFSoftShadowMap has been deprecated」と警告 → 自分で PCFShadowMap に書き換え
          // という動作をする。R3F 側が shadows を再適用するたびに PCFSoft に戻るため
          // type が往復し、そのたびに **シーン全体のマテリアルが再コンパイル**される
          // （建物が数千件あるので実害が大きい）。最初から PCFShadowMap を指定して断ち切る。
          //
          // 影の ON/OFF は enabled ではなく各オブジェクトの castShadow / receiveShadow と
          // ライトの castShadow で行う。影を落とすライトが 0 本になれば three 側が
          // シャドウパスを早期 return するので、無効化のコストも実質ゼロ。
          shadows="percentage"
          dpr={[1, 2]}
          camera={{ fov: 50, near: 0.5, far: 8000, position: [300, 320, 300] }}
          gl={{ antialias: true, powerPreference: 'high-performance' }}
          onCreated={({ gl, scene }) => {
            // 実際の色は SceneAtmosphere が毎テーマ書き直す。
            // ここは最初の 1 フレームが真っ黒に見えないようにするためだけの初期値。
            gl.setClearColor(new THREE.Color(DARK_SCENE.sky))
            scene.fog = new THREE.Fog(DARK_SCENE.sky, DARK_SCENE.fogNear, DARK_SCENE.fogFar)
          }}
        >
          {/* 空と環境光。色は日の出・日の入りで切り替わる */}
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
          {/* 最高速度標識。表示の可否と標識の一覧は SpeedSigns が自分で購読する */}
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
          {/* 認識した車線を路面へ重ねる 3D オーバーレイ。表示トグル・追従対象は
              DetectionOverlay と同じ理由で自分で購読する（ここで購読すると
              車両切り替えのたびにシーン全体が差分計算されるため） */}
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

      {/* 認識結果のオーバーレイ。カメラモード・表示トグルは自分で購読する
          （SimulatorView で購読すると車両切り替えのたびにシーン全体が差分計算されるため） */}
      <DetectionOverlay />

      <StageHud />
    </>
  )
}

// ---------------------------------------------------------------------------
// 太陽光（影を落とす平行光）
// ---------------------------------------------------------------------------

interface SunLightProps {
  cx: number
  cz: number
  extent: number
  castShadow: boolean
}

/**
 * 背景色とフォグを配色に追従させる。
 *
 * `onCreated` は 1 回しか走らないので、そこで色を決めると日の出・日の入りで
 * 切り替わらない。**フォグは差し替えずに中身を書き換える。**
 * `scene.fog` を null と非 null の間で往復させるとシーン全体のマテリアルが
 * 再コンパイルされるが、色と距離はユニフォームなので書き換えるだけなら無料。
 */
function SceneAtmosphere({ span }: { span: number }) {
  const palette = usePalette()
  const gl = useThree((s) => s.gl)
  const scene = useThree((s) => s.scene)
  const camera = useThree((s) => s.camera)

  // ★ フォグの距離は**マップの大きさに追従させる**。
  //   配色に入っている 600m / 2600m は 400m プリセット（一辺およそ 900m）に
  //   合わせた値で、金沢（広域・一辺 12.3km）にそのまま当てると
  //   **俯瞰したとき画面全体が霧一色**になり何も見えない。
  //   小さいマップでは k = 1 なので見た目は今までと変わらない。
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

  // カメラの far 面もマップに追従させる（Canvas の camera prop は生成時にしか効かない）
  useEffect(() => {
    const far = cameraFarFor(span)
    if (camera instanceof THREE.PerspectiveCamera && camera.far !== far) {
      camera.far = far
      camera.updateProjectionMatrix()
    }
  }, [camera, span])

  return null
}

/**
 * 環境光。`args` ではなく個別の props で渡す。
 * `args` は浅く比較されるので、配色が変わるたびにライトごと作り直されてしまう。
 */
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

/**
 * three の DirectionalLight は **target がシーングラフに入っていないと
 * matrixWorld が更新されない**（＝ target.position を書いても無視され、光も影カメラも
 * 常に原点を向く）。R3F の `target-position` は light.target.position に代入するだけで
 * target をシーンへ足さないので、そのままではデッドコードになる。
 * ここでは target 用の object3D を明示的にシーンへ置き、ref で light.target に差し込む。
 */
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

// ---------------------------------------------------------------------------
// ドローコールの実測（HUD 用）
// ---------------------------------------------------------------------------

/**
 * gl.info.render.calls を毎フレーム拾って sceneStats へ書く。
 *
 * three は render() の先頭で info をリセットするので、useFrame（render の前）で
 * 読むと「直前フレームの実績」が取れる。
 * 描画の重さは台数を増やしたときに黙って悪化するため、数値で見えるようにしておく。
 */
function RenderStatsProbe() {
  const gl = useThree((s) => s.gl)

  // three は render() の先頭で info をリセットする（autoReset）。
  // R3F は useFrame を render() の前に回すので、そのままだと
  // **リセット直後の 0 を読む**ことになり、HUD が常に「描画 0 call」になる。
  // 自動リセットを切って、読んだ後に自分でリセットする。
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

// ---------------------------------------------------------------------------
// 画面左下の走行状況（frameBuffer を 4Hz で覗く。60fps で React を回さない）
// ---------------------------------------------------------------------------

interface HudState {
  tick: number
  simTime: number
  active: number
  obstacles: number
  hz: number
  drawCalls: number
  notice: string
}

const EMPTY_HUD: HudState = {
  tick: 0,
  simTime: 0,
  active: 0,
  obstacles: 0,
  hz: 0,
  drawCalls: 0,
  notice: '',
}

/** 要求倍速に対してこの割合を下回ったら「追いつけていない」と見なす */
const SPEED_SHORTFALL_RATIO = 0.9

/** 表示に使う桁だけを並べた署名。これが同じなら再レンダリングしても見た目は変わらない */
function hudSignature(h: HudState): string {
  return [
    h.tick,
    h.simTime.toFixed(1),
    h.active,
    h.obstacles,
    h.hz.toFixed(1),
    h.drawCalls,
    h.notice,
  ].join('|')
}

function StageHud() {
  const mapLoaded = useSimStore((s) => s.map !== null)
  const renderPaused = useSimStore((s) => s.status.renderPaused)
  // サーバーが実際に進めているステップ速度。要求した倍速に届いているかを見る。
  // 台数と倍速の組み合わせによっては計算が間に合わず、黙って遅くなるため。
  const stepsPerSec = useSimStore((s) => s.latestMetrics?.stepsPerSec ?? 0)
  const requestedSpeed = useSimStore((s) => s.params.simSpeed)
  const simHz = useSimStore((s) => s.config.simHz)
  const actualSpeed = simHz > 0 ? stepsPerSec / simHz : 0
  const keepingUp =
    requestedSpeed <= 0 || actualSpeed >= requestedSpeed * SPEED_SHORTFALL_RATIO
  const [hud, setHud] = useState<HudState>(EMPTY_HUD)
  const lastSample = useRef({ received: 0, at: performance.now() })
  // 表示が変わらないポーリングでは setState しない（停車中に毎秒 4 回回さないため）
  const lastSignature = useRef(hudSignature(EMPTY_HUD))

  useEffect(() => {
    const timer = window.setInterval(() => {
      const curr = frameBuffer.curr
      const now = performance.now()
      const dt = (now - lastSample.current.at) / 1000
      const hz = dt > 0 ? (frameBuffer.received - lastSample.current.received) / dt : 0
      lastSample.current = { received: frameBuffer.received, at: now }

      const next: HudState = {
        tick: curr?.tick ?? 0,
        simTime: curr?.simTime ?? 0,
        active: curr ? curr.vehicles.reduce((n, v) => n + (v.active ? 1 : 0), 0) : 0,
        obstacles: frameBuffer.obstacles.length,
        hz,
        drawCalls: sceneStats.drawCalls,
        notice: currentCameraNotice(),
      }
      const sig = hudSignature(next)
      if (sig === lastSignature.current) return
      lastSignature.current = sig
      setHud(next)
    }, 250)
    return () => window.clearInterval(timer)
  }, [])

  if (!mapLoaded) return null

  return (
    <div className="app-hud">
      <div className="app-hud-item">
        走行中<span className="app-hud-value">{hud.active}</span>台
      </div>
      <div className="app-hud-item">
        経過<span className="app-hud-value">{hud.simTime.toFixed(1)}</span>秒
      </div>
      <div className="app-hud-item">
        ステップ<span className="app-hud-value">{hud.tick.toLocaleString()}</span>
      </div>
      <div className="app-hud-item">
        障害物<span className="app-hud-value">{hud.obstacles}</span>
      </div>
      <div className="app-hud-item">
        受信<span className="app-hud-value">{hud.hz.toFixed(1)}</span>Hz
      </div>
      <div
        className="app-hud-item"
        title="1 フレームあたりのドローコール数（three の gl.info.render.calls）。台数を増やしたときの描画負荷の目安"
      >
        描画<span className="app-hud-value">{hud.drawCalls}</span>call
      </div>
      <div
        className="app-hud-item"
        style={keepingUp ? undefined : { color: 'var(--m3-warning)' }}
        title={
          keepingUp
            ? '要求した倍速で進んでいます'
            : `要求 ${requestedSpeed.toFixed(2)} 倍に対して計算が間に合っていません。`
              + '車両数か倍速を下げると追いつきます（学習は止まりません）'
        }
      >
        実効<span className="app-hud-value">{actualSpeed.toFixed(2)}</span>倍
        {!keepingUp && ` / 要求 ${requestedSpeed.toFixed(2)} 倍`}
      </div>
      {hud.notice && (
        <div className="app-hud-item" style={{ color: 'var(--m3-warning)' }}>
          {hud.notice}
        </div>
      )}
      {renderPaused && (
        <div className="app-hud-item" style={{ color: 'var(--m3-warning)' }}>
          描画停止中（学習は継続）
        </div>
      )}
    </div>
  )
}
