/**
 * 3D 側から HUD へ渡す値を置く可変オブジェクト。
 *
 * frameBuffer と同じ理由で zustand には入れない。
 * ここへ書くのは 60fps の useFrame、読むのは 4Hz のポーリング（StageHud）なので、
 * React state を経由させると毎秒 60 回の再レンダリングになってしまう。
 */

export interface SceneStats {
  /** 直前フレームのドローコール数（gl.info.render.calls） */
  drawCalls: number
  /** 直前フレームの三角形数 */
  triangles: number
  /** カメラ側からユーザーへ伝えたいこと（追従対象が消えた等） */
  cameraNotice: string
  /** 上の文言を表示し続ける期限（performance.now() 基準） */
  cameraNoticeUntil: number
}

export const sceneStats: SceneStats = {
  drawCalls: 0,
  triangles: 0,
  cameraNotice: '',
  cameraNoticeUntil: 0,
}

/** 既定の表示時間 [ms]。HUD のポーリングが 4Hz なので、数秒は残さないと読めない */
const NOTICE_MS = 6000

/** カメラ側の一言を出す。同じ文言を毎フレーム書いても期限だけが伸びる */
export function setCameraNotice(text: string, ms: number = NOTICE_MS): void {
  sceneStats.cameraNotice = text
  sceneStats.cameraNoticeUntil = performance.now() + ms
}

/** 期限内なら文言、切れていれば空文字 */
export function currentCameraNotice(): string {
  if (!sceneStats.cameraNotice) return ''
  if (performance.now() > sceneStats.cameraNoticeUntil) {
    sceneStats.cameraNotice = ''
    return ''
  }
  return sceneStats.cameraNotice
}
