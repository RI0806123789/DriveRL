/** 学習済みモデルの書き出し（ダウンロード）と読み込み（アップロード）。 */

import { create } from 'zustand'

export type ExportKind = 'checkpoint' | 'torchscript' | 'pt2' | 'keras'

/** サーバーがファイル名を返さなかったときの名前 */
const FALLBACK_FILENAME: Record<ExportKind, string> = {
  checkpoint: 'autoware-sim.pt',
  torchscript: 'autoware-sim.torchscript.pt',
  pt2: 'autoware-sim.pt2',
  keras: 'autoware-sim.keras',
}

export interface ExportOutcome {
  ok: boolean
  filename?: string
  sizeBytes?: number
  error?: string
}

/** Content-Disposition ヘッダからファイル名を取り出す */
function filenameFromHeader(header: string | null, fallback: string): string {
  if (!header) return fallback
  const extended = /filename\*=UTF-8''([^;]+)/i.exec(header)
  if (extended) {
    try {
      return decodeURIComponent(extended[1])
    } catch {
    }
  }
  const plain = /filename="?([^";]+)"?/i.exec(header)
  return plain ? plain[1] : fallback
}

/** Blob をブラウザのダウンロードとして保存させる */
function saveBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  anchor.rel = 'noopener'
  document.body.appendChild(anchor)
  anchor.click()
  document.body.removeChild(anchor)
  window.setTimeout(() => URL.revokeObjectURL(url), 10_000)
}

/** モデルを書き出してダウンロードする。 */
export async function downloadModel(kind: ExportKind): Promise<ExportOutcome> {
  let response: Response
  try {
    response = await fetch(`/api/export/${kind}`, {
      method: 'GET',
      headers: { Accept: 'application/octet-stream' },
    })
  } catch (e) {
    return {
      ok: false,
      error: `サーバーに接続できませんでした（${e instanceof Error ? e.message : String(e)}）`,
    }
  }

  if (!response.ok) {
    let message = `サーバーがエラーを返しました（HTTP ${response.status}）`
    try {
      const body = (await response.json()) as { error?: string }
      if (body?.error) message = body.error
    } catch {
    }
    return { ok: false, error: message }
  }

  const filename = filenameFromHeader(
    response.headers.get('Content-Disposition'),
    FALLBACK_FILENAME[kind],
  )

  let blob: Blob
  try {
    blob = await response.blob()
  } catch (e) {
    return {
      ok: false,
      error: `ダウンロードが途中で失敗しました（${e instanceof Error ? e.message : String(e)}）`,
    }
  }

  saveBlob(blob, filename)
  return { ok: true, filename, sizeBytes: blob.size }
}

/** サーバーが返す、読み込んだチェックポイントの概要 */
export interface ImportedCheckpoint {
  updates: number
  obsDim: number
  actionDim: number
  hiddenSizes: number[]
  /** オプティマイザ状態が入っているか。無いと「続きから」の質が落ちる */
  hasOptimizer: boolean
  exportedAt: string | null
  presetId: string | null
  presetName: string | null
  metrics: Record<string, number>
}

export interface ImportOutcome {
  ok: boolean
  message?: string
  error?: string
  checkpoint?: ImportedCheckpoint
  /** 上書き前に自動保存された、それまでのモデル */
  backup?: { filename: string; sizeBytes: number } | null
}

/** 書き出したモデルをアップロードして、その状態から学習を再開する。 */
export async function importModel(file: File): Promise<ImportOutcome> {
  const form = new FormData()
  form.append('file', file, file.name)

  let response: Response
  try {
    response = await fetch('/api/import', { method: 'POST', body: form })
  } catch (e) {
    return {
      ok: false,
      error: `サーバーに接続できませんでした（${e instanceof Error ? e.message : String(e)}）`,
    }
  }

  let body: ImportOutcome
  try {
    body = (await response.json()) as ImportOutcome
  } catch {
    return { ok: false, error: `サーバーの応答を解釈できませんでした（HTTP ${response.status}）` }
  }

  if (!response.ok || !body.ok) {
    return { ok: false, error: body.error ?? `読み込みに失敗しました（HTTP ${response.status}）` }
  }
  return body
}

export interface ModelTransferState {
  /** 書き出し中の形式。null なら書き出していない */
  exporting: ExportKind | null
  exportResult: (ExportOutcome & { kind: ExportKind }) | null
  importing: boolean
  importResult: (ImportOutcome & { name: string }) | null
}

/** 書き出し・読み込みの進み具合と結果。**タブのローカル state に置かないこと** */
// タブは key で作り直されるので、最大 120 秒の読み込みの途中で移ると結果が消え、二重に送れる
export const useModelTransfer = create<ModelTransferState>(() => ({
  exporting: null,
  exportResult: null,
  importing: false,
  importResult: null,
}))

/** 書き出す。すでに書き出している間は何もしない */
export async function startExport(kind: ExportKind): Promise<void> {
  if (useModelTransfer.getState().exporting !== null) return
  useModelTransfer.setState({ exporting: kind, exportResult: null })
  try {
    const outcome = await downloadModel(kind)
    useModelTransfer.setState({ exportResult: { ...outcome, kind } })
  } finally {
    useModelTransfer.setState({ exporting: null })
  }
}

/** 読み込む。すでに読み込んでいる間は何もしない */
export async function startImport(file: File): Promise<void> {
  if (useModelTransfer.getState().importing) return
  useModelTransfer.setState({ importing: true, importResult: null })
  try {
    const outcome = await importModel(file)
    useModelTransfer.setState({ importResult: { ...outcome, name: file.name } })
  } finally {
    useModelTransfer.setState({ importing: false })
  }
}

/** バイト数を人が読める形にする */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}
