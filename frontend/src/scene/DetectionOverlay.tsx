/**
 * 認識結果（バウンディングボックス）のオーバーレイ。
 *
 * **PPO が観測として受け取っている検出結果をそのまま画面へ重ねる。** 別経路で
 * 作り直すと「画面と学習が食い違っても外から気づけない」状態になるため、
 * `frameBuffer.curr.detections` をそのまま描く（backend/app/percep/types.py の
 * docstring と同じ注意）。
 *
 * ★ 20Hz の frame は zustand に入れない約束（CLAUDE.md）。検出結果は
 *   `frameBuffer` から直接読み、`frameBuffer.received` が進んだときだけ
 *   React の state を更新する。ポーリングは `requestAnimationFrame` で行う
 *   （StageHud の setInterval と同じ「変わったときだけ setState」だが、
 *   ボックスは対象物を追って毎フレーム動くものなので間隔を詰めてある）。
 *
 * カメラモード・追従対象はこのコンポーネントが自分で購読する。
 * SimulatorView 側で購読すると、車両一覧をクリックするたびにシーン全体
 * （建物・道路・車両）の差分計算が走ってしまうため（CameraRig と同じ理由）。
 *
 * 3D オブジェクトではなく HTML/CSS の絶対配置で描く。3D 空間に置くと
 * 建物の裏に回ったときに隠れてしまい、「認識結果を見せる」目的に合わない。
 */

import { useEffect, useRef, useState, type RefObject } from 'react'
import { frameBuffer } from '../store/frameBuffer'
import { useSimStore } from '../store/simStore'
import { DET_LANE, type Detection, type FrameMessage } from '../types/protocol'
import { detectionColor, detectionLabel } from './detectionLabels'
import { projectBox } from './detectionProjection'

const EMPTY_DETECTIONS: Detection[] = []

/**
 * ラベルを枠の上ではなく内側へ回り込ませるしきい値（正規化 y0）。
 * 画面の上端に近いボックスは、上にラベルを置くと画面外へはみ出すため。
 */
const LABEL_FLIP_THRESHOLD = 0.08

/** 追従対象の車両について、いま表示すべき検出結果一覧を返すフック */
function useTrackedDetections(followTarget: number, active: boolean): Detection[] {
  const [dets, setDets] = useState<Detection[]>(EMPTY_DETECTIONS)
  const lastReceived = useRef(-1)

  useEffect(() => {
    if (!active) {
      setDets(EMPTY_DETECTIONS)
      return
    }
    // 表示 ON・対象車両の切り替え直後は、次の frame を待たず必ず 1 回読み直す
    lastReceived.current = -1
    let raf = 0
    const tick = () => {
      if (frameBuffer.received !== lastReceived.current) {
        lastReceived.current = frameBuffer.received
        const curr: FrameMessage | null = frameBuffer.curr
        setDets(curr?.detections?.[String(followTarget)] ?? EMPTY_DETECTIONS)
      }
      raf = requestAnimationFrame(tick)
    }
    tick()
    return () => cancelAnimationFrame(raf)
  }, [active, followTarget])

  return dets
}

/** キャンバスの縦横比を測る。再投影に要る（three の fov は垂直画角のため） */
function useAspect(ref: RefObject<HTMLDivElement | null>): number {
  const [aspect, setAspect] = useState(16 / 9)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const update = () => {
      const rect = el.getBoundingClientRect()
      if (rect.height > 0) setAspect(rect.width / rect.height)
    }
    update()
    const observer = new ResizeObserver(update)
    observer.observe(el)
    return () => observer.disconnect()
  }, [ref])
  return aspect
}

export function DetectionOverlay() {
  const cameraMode = useSimStore((s) => s.cameraMode)
  const followTarget = useSimStore((s) => s.followTarget)
  const showDetections = useSimStore((s) => s.view.detections)

  // ★ **運転席カメラのときだけ出す。** 擬似カメラは運転席の位置・向きで描いて
  //   いるので、追従カメラ（車体後方 15m・高さ 7.5m）に同じボックスを重ねても
  //   対象物の上には乗らない。見えている絵と無関係な枠が並ぶだけで、
  //   「何を認識しているか」をかえって誤解させる。
  const active = showDetections && cameraMode === 'driver'
  const dets = useTrackedDetections(followTarget, active)
  // コンテナは常に置く。条件付きで外すと ResizeObserver が付け外しされ、
  // 表示を戻した最初の 1 フレームだけ既定の縦横比で描いてしまう
  const ref = useRef<HTMLDivElement>(null)
  const aspect = useAspect(ref)

  return (
    <div ref={ref} className="detection-overlay">
      {active &&
        dets
          // 車線（cls===4）は矩形では道の形を表せないため、路面へ重ねる
          // LaneDetectionOverlay（3D）に置き換えた。両方出すと画面が
          // うるさくなるうえ、矩形は認識のずれを表現できず誤解を招く。
          .filter((det) => det.cls !== DET_LANE)
          // キーに配列の添字を使わない（code_review Q-12）。並びは信頼度順なので
          // 順序が入れ替わると別の検出へ DOM が使い回される。いまは DetectionBox が
          // 状態を持たないので実害は無いが、遷移を足した途端に破綻する書き方。
          .map((det, i) => (
            <DetectionBox key={`${det.cls}-${i}`} det={det} aspect={aspect} />
          ))}
    </div>
  )
}

function DetectionBox({ det, aspect }: { det: Detection; aspect: number }) {
  // 擬似カメラ（水平 68 度・4:3 固定）と three のカメラ（垂直 68 度・可変比）は
  // 画角の定義が違う。正規化座標をそのまま使うと縦に 1.27 倍ずれる
  const { left, top, width, height } = projectBox(det.box, aspect)
  // ★ 視野の外へ出た検出は描かない（code_review Q-12）。`projectBox` は 0〜1 に
  //   丸めるので、はみ出すと left と right が同じ値になり **幅 0 の枠とラベルだけが
  //   画面端に貼り付く**。キャンバスの縦横比が 1 を割ったとき（パネルを開いた
  //   縦長のウィンドウなど）に起きる。
  if (width <= 0 || height <= 0) return null
  const color = detectionColor(det)
  const label = detectionLabel(det)
  const flip = top < LABEL_FLIP_THRESHOLD

  return (
    <div
      className="detection-box"
      style={{
        left: `${left * 100}%`,
        top: `${top * 100}%`,
        width: `${width * 100}%`,
        height: `${height * 100}%`,
        borderColor: color,
      }}
    >
      <span
        className="detection-label"
        style={flip ? { top: '2px' } : { bottom: 'calc(100% + 3px)' }}
      >
        <span className="detection-label-dot" style={{ background: color }} />
        {label}
        {det.distance !== undefined && (
          <span className="detection-label-distance">{det.distance.toFixed(1)}m</span>
        )}
      </span>
    </div>
  )
}
