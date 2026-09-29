/** オンライン模倣のアシスト率（metrics.assistRate）と onlineAssist の契約を検証する（ブラウザ不要）。 */

import { readFileSync } from 'node:fs'

import { ASSIST_P_MIN, assistRateView, normalizeAssistRate } from '../src/store/assistRate.ts'

let failures = 0

function check(label: string, ok: boolean, detail = ''): void {
  console.log(`  [${ok ? 'OK  ' : 'NG  '}] ${label}${detail ? ` — ${detail}` : ''}`)
  if (!ok) failures++
}

console.log('1. 値の正規化（0.0〜1.0 に収め、数でなければ null）')
const cases: Array<[unknown, number | null]> = [
  [0, 0],
  [1, 1],
  [0.85, 0.85],
  [-0.01, 0],
  [1.0001, 1],
  [Number.NaN, null],
  [Number.POSITIVE_INFINITY, null],
  [undefined, null],
  [null, null],
  ['0.5', null],
]
for (const [input, want] of cases) {
  const got = normalizeAssistRate(input)
  check(`normalizeAssistRate(${String(input)}) = ${String(want)}`, got === want, `実際 ${String(got)}`)
}

console.log('\n2. 表示の文言と色')
check('0.85 は 85%', assistRateView(0.85).text === 'Assist Rate 85%')
check('0.004 は 0%（切り上げない）', assistRateView(0.004).percent === 0)
check('0.995 は 100%', assistRateView(0.995).percent === 100)
check('届いていなければ percent は null', assistRateView(undefined).percent === null)
check('下限まで下がれば ok の色', assistRateView(ASSIST_P_MIN).tone === 'ok')
check('半分以上なら warning の色', assistRateView(0.5).tone === 'warning')
{
  let monotone = true
  let prev = -1
  for (let i = 0; i <= 1000; i++) {
    const p = assistRateView(i / 1000).percent ?? -1
    if (p < prev) monotone = false
    prev = p
  }
  check('percent は値に対して単調に増える', monotone)
}

console.log('\n3. 契約ファイルとの突き合わせ')
const protocolTs = readFileSync('src/types/protocol.ts', 'utf8')
const protocolMd = readFileSync('../docs/protocol.md', 'utf8')
const assistPy = readFileSync('../backend/app/rl/online_assist.py', 'utf8')
const contractsPy = readFileSync('../backend/app/contracts.py', 'utf8')
const simStore = readFileSync('src/store/simStore.ts', 'utf8')

function interfaceBody(name: string): string {
  const m = new RegExp(`export interface ${name} \\{([\\s\\S]*?)\\n\\}`).exec(protocolTs)
  return m?.[1] ?? ''
}
check('MetricsMessage に assistRate?: number がある', /\n  assistRate\?: number/.test(interfaceBody('MetricsMessage')))
check('MetricsMessage に bcLoss?: number がある', /\n  bcLoss\?: number/.test(interfaceBody('MetricsMessage')))
check('SimParams に onlineAssist: boolean がある', /\n  onlineAssist: boolean/.test(interfaceBody('SimParams')))
check('docs/protocol.md の metrics の例に assistRate がある', /"assistRate":/.test(protocolMd))
check('docs/protocol.md の params の例に onlineAssist がある', /"onlineAssist":/.test(protocolMd))

const pyMin = /^ASSIST_P_MIN = ([\d.]+)/m.exec(assistPy)?.[1]
check(
  'ASSIST_P_MIN がバックエンドと一致する',
  pyMin !== undefined && Number(pyMin) === ASSIST_P_MIN,
  `backend ${pyMin} / frontend ${ASSIST_P_MIN}`,
)
const pyDefault = /^\s+online_assist: bool = (True|False)/m.exec(contractsPy)?.[1]
const tsDefault = /onlineAssist: (true|false),/.exec(simStore)?.[1]
check(
  'onlineAssist の既定値がバックエンドと一致する',
  pyDefault !== undefined && tsDefault !== undefined && (pyDefault === 'True') === (tsDefault === 'true'),
  `backend ${pyDefault} / frontend ${tsDefault}`,
)

console.log('')
if (failures > 0) {
  console.log(`NG: ${failures} 件`)
  process.exit(1)
}
console.log('OK: すべての検査に通りました')
