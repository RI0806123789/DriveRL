/** ヒヤリハットのオートカリキュラム（`metrics.curriculumLevel` など）の正規化と表示の形。 */

/** 難易度 1.0 のときの発生確率（`backend/app/sim/curriculum.py` の `INCIDENT_MAX_PROB`） */
export const INCIDENT_MAX_PROB = 0.2
/** 難易度の刻み（同じく `LEVEL_STEP`） */
export const LEVEL_STEP = 0.05

/** 画面の表示の形 */
export interface CurriculumView {
  /** ゲージの伸び 0〜1（`--m3-bar-value` に渡す）。値が届いていなければ 0 */
  gauge: number
  levelText: string
  probabilityText: string
  triggeredText: string
  avoidedText: string
}

/** 0〜1 の割合に収める。数でない・届いていないときは null。 */
export function normalizeRatio(value: unknown): number | null {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null
  return Math.min(1, Math.max(0, value))
}

function normalizeCount(value: unknown): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) return 0
  return Math.floor(value)
}

/** 難易度・件数・回避率から表示の文言を作る。件数 0 のときは回避率を「—」にする（0 で割らない）。 */
export function curriculumView(level: unknown, triggered: unknown, avoided: unknown): CurriculumView {
  const value = normalizeRatio(level)
  const count = normalizeCount(triggered)
  const rate = normalizeRatio(avoided)
  const percent = value === null ? null : Math.round(value * 100)
  const probability = value === null ? null : value * INCIDENT_MAX_PROB * 100
  return {
    gauge: value ?? 0,
    levelText: percent === null ? 'Curriculum Level —' : `Curriculum Level ${percent}%`,
    probabilityText: probability === null ? '—' : `${probability.toFixed(0)}%`,
    triggeredText: `${count.toLocaleString()} 回`,
    avoidedText: count === 0 || rate === null ? '—' : `${(rate * 100).toFixed(1)}%`,
  }
}

/** モックが返す値。実機と同じく難易度は刻みごとに上がる。乱数は引かない。 */
export function mockCurriculum(
  enabled: boolean,
  progress: number,
  updates: number,
): { curriculumLevel: number; incidentsTriggered: number; incidentsAvoidedRate: number | null } {
  if (!enabled) return { curriculumLevel: 0, incidentsTriggered: 0, incidentsAvoidedRate: null }
  const raw = Math.min(1, Math.max(0, (progress - 0.4) / 0.6))
  const level = Math.round(raw / LEVEL_STEP) * LEVEL_STEP
  const triggered = Math.floor(updates * level * 0.5)
  return {
    curriculumLevel: level,
    incidentsTriggered: triggered,
    incidentsAvoidedRate: triggered > 0 ? 0.55 + 0.35 * progress : null,
  }
}
