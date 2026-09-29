/** ヒヤリハットのオートカリキュラムの表示の算術と、契約ファイル・バックエンドの定数との突き合わせを検証する（ブラウザ不要）。 */

import { readFileSync } from 'node:fs'

import {
  INCIDENT_MAX_PROB,
  LEVEL_STEP,
  curriculumView,
  mockCurriculum,
  normalizeRatio,
} from '../src/store/curriculum.ts'

let failures = 0

function check(label: string, ok: boolean, detail = ''): void {
  console.log(`  [${ok ? 'OK  ' : 'NG  '}] ${label}${detail ? ` — ${detail}` : ''}`)
  if (!ok) failures++
}

console.log('1. 割合の正規化（0.0〜1.0 に収め、数でなければ null）')
const cases: Array<[unknown, number | null]> = [
  [0, 0],
  [1, 1],
  [0.65, 0.65],
  [-0.5, 0],
  [3, 1],
  [Number.NaN, null],
  [Number.NEGATIVE_INFINITY, null],
  [undefined, null],
  [null, null],
  ['0.3', null],
]
for (const [input, want] of cases) {
  const got = normalizeRatio(input)
  check(`normalizeRatio(${String(input)}) = ${String(want)}`, got === want, `実際 ${String(got)}`)
}

console.log('\n2. ゲージと文言')
{
  const v = curriculumView(0.65, 20, 0.9)
  check('0.65 はゲージ 0.65・Curriculum Level 65%', v.gauge === 0.65 && v.levelText === 'Curriculum Level 65%')
  check('発生確率は L × 20%（0.65 → 13%）', v.probabilityText === '13%', v.probabilityText)
  check('回避率は小数 1 桁', v.avoidedText === '90.0%', v.avoidedText)
}
{
  const zero = curriculumView(0.4, 0, 0.5)
  check('発生件数 0 なら回避率は —（0 で割らない）', zero.avoidedText === '—', zero.avoidedText)
  const pending = curriculumView(0.4, 5, null)
  check('件数があっても未確定（null）なら —', pending.avoidedText === '—')
  const none = curriculumView(undefined, undefined, undefined)
  const text = JSON.stringify(none)
  check('何も届いていなくても NaN を出さない', !text.includes('NaN'), text)
  check('何も届いていなければゲージは 0', none.gauge === 0)
  check('負の件数や小数の件数は 0 以上の整数にする', curriculumView(0.1, -3, 0.5).triggeredText === '0 回' && curriculumView(0.1, 2.7, 0.5).triggeredText === '2 回')
}
{
  let monotone = true
  let prev = -1
  for (let i = 0; i <= 1000; i++) {
    const g = curriculumView(i / 1000, 1, 1).gauge
    if (g < prev) monotone = false
    prev = g
  }
  check('ゲージは難易度に対して単調に伸びる', monotone)
}

console.log('\n3. モック')
{
  const off = mockCurriculum(false, 1, 1000)
  check('切っていれば 0 件・難易度 0・回避率 null', off.curriculumLevel === 0 && off.incidentsTriggered === 0 && off.incidentsAvoidedRate === null)
  const early = mockCurriculum(true, 0.2, 100)
  check('学習の始めは難易度 0・回避率 null', early.curriculumLevel === 0 && early.incidentsAvoidedRate === null)
  let steps = true
  let rising = true
  let prev = 0
  for (let i = 0; i <= 100; i++) {
    const m = mockCurriculum(true, i / 100, 500)
    const ratio = m.curriculumLevel / LEVEL_STEP
    if (Math.abs(ratio - Math.round(ratio)) > 1e-9) steps = false
    if (m.curriculumLevel < prev) rising = false
    prev = m.curriculumLevel
  }
  check('難易度は刻み（0.05）ごとの値だけ', steps)
  check('学習が進むほど難易度は下がらない', rising)
}

console.log('\n4. 契約ファイルとバックエンドの定数')
const protocolTs = readFileSync('src/types/protocol.ts', 'utf8')
const protocolMd = readFileSync('../docs/protocol.md', 'utf8')
const curriculumPy = readFileSync('../backend/app/sim/curriculum.py', 'utf8')
const contractsPy = readFileSync('../backend/app/contracts.py', 'utf8')
const simStore = readFileSync('src/store/simStore.ts', 'utf8')

function interfaceBody(name: string): string {
  return new RegExp(`export interface ${name} \\{([\\s\\S]*?)\\n\\}`).exec(protocolTs)?.[1] ?? ''
}
const metrics = interfaceBody('MetricsMessage')
check('MetricsMessage に curriculumLevel?: number がある', /\n  curriculumLevel\?: number\n/.test(metrics))
check('MetricsMessage に incidentsTriggered?: number がある', /\n  incidentsTriggered\?: number\n/.test(metrics))
check('MetricsMessage に incidentsAvoidedRate?: number | null がある', /\n  incidentsAvoidedRate\?: number \| null/.test(metrics))
check('SimParams に incidentCurriculum: boolean がある', /\n  incidentCurriculum: boolean/.test(interfaceBody('SimParams')))
for (const key of ['curriculumLevel', 'incidentsTriggered', 'incidentsAvoidedRate', 'incidentCurriculum']) {
  check(`docs/protocol.md の例に ${key} がある`, new RegExp(`"${key}":`).test(protocolMd))
}
const pyNumber = (name: string) => Number(new RegExp(`^${name} = ([\\d.]+)`, 'm').exec(curriculumPy)?.[1])
check('INCIDENT_MAX_PROB がバックエンドと一致する', pyNumber('INCIDENT_MAX_PROB') === INCIDENT_MAX_PROB, `backend ${pyNumber('INCIDENT_MAX_PROB')}`)
check('LEVEL_STEP がバックエンドと一致する', pyNumber('LEVEL_STEP') === LEVEL_STEP, `backend ${pyNumber('LEVEL_STEP')}`)
const pyDefault = /^\s+incident_curriculum: bool = (True|False)/m.exec(contractsPy)?.[1]
const tsDefault = /incidentCurriculum: (true|false),/.exec(simStore)?.[1]
check(
  'incidentCurriculum の既定値がバックエンドと一致する',
  pyDefault !== undefined && tsDefault !== undefined && (pyDefault === 'True') === (tsDefault === 'true'),
  `backend ${pyDefault} / frontend ${tsDefault}`,
)

console.log('')
if (failures > 0) {
  console.log(`NG: ${failures} 件`)
  process.exit(1)
}
console.log('OK: すべての検査に通りました')
