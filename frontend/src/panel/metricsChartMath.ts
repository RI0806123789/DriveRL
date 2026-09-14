/**
 * 学習指標グラフの計算。React から切り離した純粋モジュール。
 *
 * ここを React の中に置かないのは、`npm run verify` の幾何検証と同じ理由:
 * **間引きの間違いは型チェックもビルドも通る**。目盛りの「最小 / 最大」と
 * 実際に描かれた線が食い違っても、グラフとして成立してしまうので気づけない。
 *
 * ★ 点数に上限は無い。simStore がメトリクスを間引かずに全部持つので
 *   （学習の始めから終わりまでを 1 枚で見たいため）、1Hz × 走らせた時間だけ増える。
 *   1 時間で 3,600 点、8 時間で 28,800 点。したがってここは
 *   **「点の数に比例して重くならない」ことを守る**:
 *
 *   1. 統計は 1 パスのループで出す。`Math.min(...values)` は使わない
 *      （引数が約 65,000 個を超えると RangeError でグラフが丸ごと消える。
 *       全データを出す以上、18 時間ほど回せば必ず踏む）
 *   2. パスは MAX_BUCKETS 本に間引く。viewBox の幅が 300 なので、
 *      それ以上打っても画面には出ないのに文字列だけが伸びる
 *      （8 時間分をそのまま打つと、毎秒 20 万個のパスコマンドを組み立てることになる）
 */

/** SVG の viewBox の横幅 */
export const VIEW_W = 300

/**
 * 横方向に何本まで打つか。viewBox の幅と同じにしてある（1 本 = 1px 相当）。
 * 1 本につき最小・最大の 2 点を出すので、パスの点数は最大でもこの 2 倍。
 */
export const MAX_BUCKETS = VIEW_W

/** 1 本ぶんの縦の範囲。min/max を出た順に持つ（描く順を時系列に合わせるため） */
interface Bucket {
  min: number
  max: number
  /** min と max のどちらが先に来たか。true なら min が先 */
  minFirst: boolean
}

export interface ChartGeometry {
  /** 折れ線の d 属性 */
  line: string
  /** 塗り用に下端まで閉じた d 属性 */
  area: string
  /** 全データの最小・最大・平均（★ 間引く前から出す） */
  min: number
  max: number
  mean: number
  /** 最後の有限値 */
  latest: number
  /** 有限値の個数 */
  count: number
  /** 0 の基準線の y。範囲に 0 が入らなければ null */
  zeroY: number | null
}

export interface ChartOptions {
  height: number
  /** 0 を必ず範囲に入れる（報酬など正負をまたぐ指標で有効） */
  showZero?: boolean
}

/**
 * 値の列から SVG のパスと目盛りを作る。
 * 有限値が 2 点に満たなければ null（グラフとして意味が無い）。
 */
export function buildChart(
  values: readonly number[],
  { height, showZero = false }: ChartOptions,
): ChartGeometry | null {
  const n = values.length

  // --- 1 パス目: 統計。ここで配列のコピーを作らない ---
  let min = Infinity
  let max = -Infinity
  let sum = 0
  let count = 0
  let latest = NaN
  for (let i = 0; i < n; i++) {
    const v = values[i]
    if (!Number.isFinite(v)) continue
    if (v < min) min = v
    if (v > max) max = v
    sum += v
    count += 1
    latest = v
  }
  if (count < 2) return null

  // 平均は**全データ**から出す。下の間引きは描画用なので、
  // 間引いた後の点から平均を取ると値がずれる
  const mean = sum / count

  // --- 目盛りの範囲 ---
  let lo = min
  let hi = max
  if (showZero) {
    lo = Math.min(lo, 0)
    hi = Math.max(hi, 0)
  }
  // 平坦な系列でも線が真ん中に出るように、幅ゼロを避ける
  if (hi - lo < 1e-9) {
    const pad = Math.max(1e-6, Math.abs(hi) * 0.1)
    lo -= pad
    hi += pad
  }
  const span = hi - lo
  const toY = (v: number) => height - ((v - lo) / span) * height

  // --- 2 パス目: 横方向に間引く ---
  // バケツ分けは**有限値の並び順**で行う（欠測があっても横に隙間ができない）。
  const buckets = Math.min(MAX_BUCKETS, count)
  const slots: (Bucket | null)[] = new Array(buckets).fill(null)
  let seen = 0
  for (let i = 0; i < n; i++) {
    const v = values[i]
    if (!Number.isFinite(v)) continue
    // count と buckets から決まるので、最後の点は必ず最後のバケツに入る
    const b = Math.min(buckets - 1, Math.floor((seen * buckets) / count))
    seen += 1
    const cur = slots[b]
    if (cur === null) {
      slots[b] = { min: v, max: v, minFirst: true }
    } else if (v < cur.min) {
      cur.min = v
      cur.minFirst = false // 後から下を更新した = 先に来ていた max のほうが前
    } else if (v > cur.max) {
      cur.max = v
      // ★ ここを書き忘れない。後から上を更新したなら min のほうが前になる。
      //   5 → 3 → 7 の順で来ると minFirst が false のまま残り、
      //   縦線を描く向きが逆になる（見た目では気づけない）
      cur.minFirst = true
    }
  }

  // --- パスを組む ---
  const stepX = buckets > 1 ? VIEW_W / (buckets - 1) : 0
  let line = ''
  for (let b = 0; b < buckets; b++) {
    const s = slots[b]
    if (s === null) continue
    const x = (b * stepX).toFixed(2)
    // 1 本の中で上下に振れている分は縦線として残す。
    // 平均で潰すと、スパイク（報酬が一度だけ跳ねた等）が消えて
    // 下に出る「最小 / 最大」と絵が食い違う
    const first = s.minFirst ? s.min : s.max
    const second = s.minFirst ? s.max : s.min
    line += `${line === '' ? 'M' : 'L'}${x},${toY(first).toFixed(2)}`
    if (second !== first) line += `L${x},${toY(second).toFixed(2)}`
  }

  return {
    line,
    area: `${line}L${VIEW_W},${height}L0,${height}Z`,
    min,
    max,
    mean,
    latest,
    count,
    zeroY: lo <= 0 && hi >= 0 ? toY(0) : null,
  }
}
