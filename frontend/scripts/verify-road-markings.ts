/**
 * 道路標示の検証（ブラウザ不要）。
 *
 *   node scripts/verify-road-markings.ts
 *
 * 破線の割り付けは「引かれない」という形で壊れる。画面を見ても
 * 「その道路には中央線が無い」としか見えず、型チェックもビルドも通ってしまうので
 * 数値で確かめる（code_review S-03）。
 */

import {
  DASH_OFF,
  DASH_ON,
  dashSpans,
} from '../src/scene/roadMarkingGeometry.ts'

let failed = 0

function check(label: string, ok: boolean, detail = ''): void {
  if (!ok) failed++
  console.log(`  [${ok ? 'OK  ' : 'NG  '}] ${label}${detail ? ` — ${detail}` : ''}`)
}

const PERIOD = DASH_ON + DASH_OFF

console.log('')
console.log('='.repeat(70))
console.log('道路標示: 破線の割り付け')
console.log('='.repeat(70))
console.log('')

console.log('1. 短い道路にも必ず 1 本引かれる（S-03 の回帰）')
{
  // 以前はここが 0 本だった。境目はちょうど DASH_ON = 5m
  for (const total of [0.6, 1.0, 2.5, 4.0, 4.99]) {
    const spans = dashSpans(total)
    check(
      `全長 ${total}m でも破線が引かれる`,
      spans.length === 1,
      `${spans.length} 本`,
    )
    if (spans.length === 1) {
      check(
        `  その線が道路に収まる`,
        spans[0].start >= 0 && spans[0].end <= total + 1e-9,
        `${spans[0].start.toFixed(2)}〜${spans[0].end.toFixed(2)}m`,
      )
    }
  }
}

console.log('')
console.log('2. 以前から正しかった長さの挙動が変わっていない')
{
  // 5m 以上では余白が非負なので、修正前と同じ結果でなければならない
  const legacy = (total: number) => {
    const count = Math.max(1, Math.floor(total / PERIOD))
    const margin = (total - count * PERIOD + DASH_OFF) / 2
    const out: { start: number; end: number }[] = []
    for (let k = 0; k < count; k++) {
      const start = margin + k * PERIOD
      const end = start + DASH_ON
      if (start < 0 || end > total) continue
      out.push({ start, end })
    }
    return out
  }
  let worst = 0
  let sameCount = true
  for (let total = 5; total <= 400; total += 0.5) {
    const a = legacy(total)
    const b = dashSpans(total)
    if (a.length !== b.length) {
      sameCount = false
      break
    }
    for (let i = 0; i < a.length; i++) {
      worst = Math.max(worst, Math.abs(a[i].start - b[i].start), Math.abs(a[i].end - b[i].end))
    }
  }
  check('5m 以上では修正前と本数が同じ', sameCount)
  check('5m 以上では修正前と位置も同じ', worst === 0, `最大差 ${worst}`)
}

console.log('')
console.log('3. 割り付けの不変条件')
{
  let ok = true
  let centered = true
  let ordered = true
  for (let total = 0.5; total <= 500; total += 0.25) {
    const spans = dashSpans(total)
    if (spans.length < 1) ok = false
    for (let i = 0; i < spans.length; i++) {
      const s = spans[i]
      if (s.start < -1e-9 || s.end > total + 1e-9 || s.end <= s.start) ok = false
      if (i > 0 && s.start < spans[i - 1].end) ordered = false
    }
    // 前後の余白が等しい（総長が線部より長い場合）
    if (total >= DASH_ON) {
      const head = spans[0].start
      const tail = total - spans[spans.length - 1].end
      if (Math.abs(head - tail) > 1e-9) centered = false
    }
  }
  check('どの長さでも 1 本以上引かれ、道路からはみ出さない', ok)
  check('破線どうしが重ならない', ordered)
  check('前後の余白が等しい（5m 以上）', centered)
}

console.log('')
console.log('4. 縮退した入力')
{
  check('全長 0 なら引かない', dashSpans(0).length === 0)
  check('負の全長なら引かない', dashSpans(-3).length === 0)
  check('NaN なら引かない', dashSpans(Number.NaN).length === 0)
}

console.log('')
console.log('='.repeat(70))
console.log(failed === 0 ? '結果: すべて合格' : `結果: ${failed} 件 失敗`)
console.log('='.repeat(70))
process.exit(failed === 0 ? 0 : 1)
