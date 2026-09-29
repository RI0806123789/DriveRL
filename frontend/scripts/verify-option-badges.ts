/** 階層型の方策の意図のバッジ（配色・アイコン・クラス）の排他性と、配色トークン・バックエンドの定数・契約ファイルとの突き合わせを検証する（ブラウザ不要）。 */

import { readFileSync } from 'node:fs'

import {
  DRIVE_OPTION_STYLES,
  jerkText,
  normalizeDriveOption,
  optionShareRows,
} from '../src/store/driveOption.ts'
import { MOCK_OPTION_STEPS, MockOptions, guessOption, mockOptionMetrics } from '../src/store/mock/options.ts'
import { DRIVE_OPTIONS } from '../src/types/protocol.ts'
import type { VehicleState } from '../src/types/protocol.ts'

let failures = 0

function check(label: string, ok: boolean, detail = ''): void {
  console.log(`  [${ok ? 'OK  ' : 'NG  '}] ${label}${detail ? ` — ${detail}` : ''}`)
  if (!ok) failures++
}

const root = new URL('../', import.meta.url)
const read = (path: string): string => readFileSync(new URL(path, root), 'utf8')

console.log('1. バックエンドの定数と揃っているか')
{
  const config = read('../backend/app/config.py')
  const names = /HRL_OPTIONS = \(([^)]*)\)/.exec(config)?.[1].match(/"([A-Z]+)"/g)?.map((s) => s.slice(1, -1)) ?? []
  check('意図の名前と並びが config.HRL_OPTIONS と同じ', names.join(',') === DRIVE_OPTIONS.join(','), names.join(','))
  const steps = Number(/HRL_OPTION_STEPS = (\d+)/.exec(config)?.[1])
  check('モックの保つ長さが config.HRL_OPTION_STEPS と同じ', steps === MOCK_OPTION_STEPS, `${steps}`)
}

console.log('2. 4 つの意図の見た目が互いに排他か')
{
  const styles = DRIVE_OPTIONS.map((o) => DRIVE_OPTION_STYLES[o])
  check('どの意図にも見た目がある（キーと中身の名前が一致）', styles.every((s, i) => s && s.option === DRIVE_OPTIONS[i]))
  for (const key of ['className', 'token', 'hue', 'icon', 'label'] as const) {
    const values = styles.map((s) => s[key])
    check(`${key} が重ならない`, new Set(values).size === values.length, values.join(' / '))
  }
  const hues = DRIVE_OPTIONS.map((o) => `${o}=${DRIVE_OPTION_STYLES[o].hue}`).join(' ')
  check(
    'CRUISE 青 / FOLLOW 緑 / YIELD 黄 / STOP 赤',
    hues === 'CRUISE=blue FOLLOW=green YIELD=yellow STOP=red',
    hues,
  )
  check('クラス名は option-badge--<小文字の名前>', styles.every((s) => s.className === `option-badge--${s.option.toLowerCase()}`))
}

console.log('3. 配色トークンが昼夜の両方に定義され、読めるコントラストか')
{
  const tokens = read('src/styles/tokens.css')
  const lightAt = tokens.indexOf(":root[data-theme='light']")
  const blocks = { 夜: tokens.slice(0, lightAt), 昼: tokens.slice(lightAt) }
  const hex = (block: string, name: string): string | null =>
    new RegExp(`${name}:\\s*(#[0-9a-fA-F]{6})`).exec(block)?.[1] ?? null
  const luminance = (c: string): number => {
    const ch = [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16) / 255).map((v) => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4))
    return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2]
  }
  for (const [theme, block] of Object.entries(blocks)) {
    const backs: string[] = []
    for (const option of DRIVE_OPTIONS) {
      const style = DRIVE_OPTION_STYLES[option]
      const back = hex(block, style.token)
      const fore = hex(block, style.token.replace('--m3-option-', '--m3-on-option-'))
      if (!back || !fore) {
        check(`${theme}: ${option} の地と文字の色がある`, false)
        continue
      }
      backs.push(back.toLowerCase())
      const [a, b] = [luminance(back), luminance(fore)].sort((x, y) => y - x)
      const ratio = (a + 0.05) / (b + 0.05)
      check(`${theme}: ${option} の文字と地のコントラストが 4.5 以上`, ratio >= 4.5, ratio.toFixed(1))
    }
    check(`${theme}: 4 つの地の色が重ならない`, new Set(backs).size === DRIVE_OPTIONS.length, backs.join(' '))
  }
  const css = read('src/styles/global.css')
  for (const option of DRIVE_OPTIONS) {
    const style = DRIVE_OPTION_STYLES[option]
    const rule = new RegExp(`\\.${style.className}\\s*\\{[^}]*background:\\s*var\\(${style.token}\\)`).exec(css)
    check(`global.css の .${style.className} が ${style.token} を地にする`, rule !== null)
  }
}

console.log('4. 受け取った値の扱い')
{
  check('4 つの名前は通す', DRIVE_OPTIONS.every((o) => normalizeDriveOption(o) === o))
  check('知らない名前・小文字・数・未定義は null', ['TURBO', 'stop', '', 2, undefined, null].every((v) => normalizeDriveOption(v) === null))
  const rows = optionShareRows([2, 1, 1, 0])
  check('割合は合計 1 に正規化して DRIVE_OPTIONS の順に並べる', rows !== null && rows.map((r) => r.style.option).join(',') === DRIVE_OPTIONS.join(',') && Math.abs(rows.reduce((a, r) => a + r.share, 0) - 1) < 1e-9 && rows[0].percent === 50)
  check('長さ違い・NaN・合計 0・未定義は null', [[1, 2, 3], [1, Number.NaN, 1, 1], [0, 0, 0, 0], undefined, 'x'].every((v) => optionShareRows(v) === null))
  check('負の値は 0 として扱う', optionShareRows([-1, 1, 1, 2])?.[0].share === 0)
  check('加加速度は小数 1 桁・無ければ —', jerkText(3.456) === '3.5 m/s³' && jerkText(null) === '—' && jerkText(Number.NaN) === '—')
}

console.log('5. モックの意図（20 ステップ保つ・乱数を引かない）')
{
  const car = (id: number, speed: number, braking = false, active = true): VehicleState =>
    ({ id, active, speed, braking } as unknown as VehicleState)
  check('止まっていれば STOP・減速中で遅ければ YIELD・減速中なら FOLLOW・それ以外は CRUISE',
    guessOption(car(0, 0)) === 'STOP' && guessOption(car(0, 2, true)) === 'YIELD' && guessOption(car(0, 8, true)) === 'FOLLOW' && guessOption(car(0, 8)) === 'CRUISE')
  const m = new MockOptions()
  const seen: string[] = []
  const changes: number[] = []
  for (let tick = 1; tick <= 80; tick++) {
    const speed = tick % 7 === 0 ? 0 : 9
    const [v] = m.apply([car(0, speed)], tick)
    if (seen.length && seen[seen.length - 1] !== v.currentOption) changes.push(tick)
    seen.push(v.currentOption ?? '')
  }
  check('意図が変わるのは 20 ステップごとの選び直しのときだけ', changes.every((t) => t % MOCK_OPTION_STEPS === 0), changes.join(','))
  const [off] = m.apply([car(0, 9, false, false)], 81)
  check('走っていない車には載せない', off.currentOption === undefined)
  const early = mockOptionMetrics(0)
  const late = mockOptionMetrics(1)
  check('ダミーの割合は合計 1', Math.abs(early.optionShares.reduce((a, b) => a + b, 0) - 1) < 1e-9 && early.optionShares.length === 4)
  check('学習が進むと STOP が減り、加加速度も下がる', late.optionShares[3] < early.optionShares[3] && late.jerkRms < early.jerkRms)
}

console.log('6. 契約ファイル')
{
  const doc = read('../docs/protocol.md')
  for (const key of ['currentOption', 'optionShares', 'jerkRms']) {
    check(`docs/protocol.md に ${key} の説明がある`, doc.includes(key))
  }
  const ts = read('src/types/protocol.ts')
  check('protocol.ts の FrameVehicle に currentOption?: DriveOption', /\n  currentOption\?: DriveOption\n/.test(ts))
}

if (failures > 0) {
  console.log(`\nNG: ${failures} 件`)
  process.exit(1)
}
console.log('\nすべて OK')
