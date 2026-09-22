/** コメント規約と CSS の遷移規約を検査する（CLAUDE.md「コードの書き方」）。 */

import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join } from 'node:path'

let failures = 0

function check(label: string, ok: boolean, detail = ''): void {
  console.log(`  [${ok ? 'OK  ' : 'NG  '}] ${label}${detail ? ` — ${detail}` : ''}`)
  if (!ok) failures += 1
}

/** 印。**この検証スクリプト自身と CLAUDE.md 以外では使わない** */
const MARK = String.fromCodePoint(0x2605)

interface Target {
  readonly dir: string
  readonly exts: readonly string[]
  readonly recursive: boolean
}

const TARGETS: Target[] = [
  { dir: '../backend/app', exts: ['.py'], recursive: true },
  { dir: '../backend', exts: ['.py'], recursive: false },
  { dir: 'src', exts: ['.ts', '.tsx'], recursive: true },
  { dir: 'scripts', exts: ['.ts'], recursive: true },
]

function collect(t: Target, out: string[] = []): string[] {
  for (const name of readdirSync(t.dir)) {
    const p = join(t.dir, name)
    const st = statSync(p)
    if (st.isDirectory()) {
      if (t.recursive && name !== '__pycache__' && name !== 'node_modules') {
        collect({ ...t, dir: p }, out)
      }
      continue
    }
    if (t.exts.some((e) => name.endsWith(e))) out.push(p)
  }
  return out
}

const files: string[] = []
for (const t of TARGETS) files.push(...collect(t))
const unique = [...new Set(files)]

console.log(`対象 ${unique.length} ファイル（backend/app・backend・src・scripts）`)

console.log('')
console.log(`コードに ${MARK} を書かない（設計の意図は CLAUDE.md 側へ）`)
{
  const hits: Array<[string, number]> = []
  for (const f of unique) {
    if (f.endsWith('verify-conventions.ts')) continue
    const n = readFileSync(f, 'utf8').split(MARK).length - 1
    if (n > 0) hits.push([f, n])
  }
  const total = hits.reduce((a, b) => a + b[1], 0)
  check(`コード中の ${MARK} は 0 件`, total === 0, `${total} 件 / ${hits.length} ファイル`)
  for (const [f, n] of hits.slice(0, 10)) console.log(`        ${f}: ${n} 件`)
}

console.log('')
console.log('docstring / JSDoc は 1 行（複数行は CLAUDE.md へ移すこと）')
{
  const multiPy: Array<[string, number]> = []
  const multiTs: Array<[string, number]> = []
  for (const f of unique) {
    const lines = readFileSync(f, 'utf8').split(/\r?\n/)
    if (f.endsWith('.py')) {
      for (let i = 0; i < lines.length; i += 1) {
        const t = lines[i].trim()
        if (!t.startsWith('"""')) continue
        // 同じ行で閉じていれば 1 行
        if (t.length > 3 && t.endsWith('"""')) continue
        multiPy.push([f, i + 1])
        let j = i + 1
        while (j < lines.length && !lines[j].includes('"""')) j += 1
        i = j
      }
    } else {
      for (let i = 0; i < lines.length; i += 1) {
        const t = lines[i].trim()
        if (!t.startsWith('/**')) continue
        if (t.endsWith('*/') && t.length > 4) continue
        multiTs.push([f, i + 1])
        let j = i
        while (j < lines.length && !lines[j].trim().endsWith('*/')) j += 1
        i = j
      }
    }
  }
  check('Python の複数行 docstring は 0 件', multiPy.length === 0, `${multiPy.length} 件`)
  for (const [f, l] of multiPy.slice(0, 10)) console.log(`        ${f}:${l}`)
  check('TypeScript の複数行 /** */ は 0 件', multiTs.length === 0, `${multiTs.length} 件`)
  for (const [f, l] of multiTs.slice(0, 10)) console.log(`        ${f}:${l}`)
}

console.log('')
console.log('CSS はレイアウトを起こすプロパティを遷移させない')
{
  const ALLOWED = new Set([
    'transform',
    'opacity',
    'background-color',
    'color',
    'border-color',
    'outline-color',
    'border-radius',
    'visibility',
  ])
  const css = readFileSync('src/styles/global.css', 'utf8')
  const bad: Array<[number, string]> = []
  let count = 0
  for (const m of css.matchAll(/(^|[\s;{])transition:\s*([^;}]*)[;}]/g)) {
    const value = m[2].replace(/\s+/g, ' ').trim()
    if (value === 'none') continue
    const line = css.slice(0, m.index).split('\n').length
    for (const part of value.split(',')) {
      const prop = part.trim().split(/\s+/)[0]
      if (!prop) continue
      count += 1
      if (!ALLOWED.has(prop)) bad.push([line, prop])
    }
  }
  check(
    '遷移させているのは transform / opacity と塗り直しだけのものに限る',
    bad.length === 0,
    `${count} プロパティ中 ${bad.length} 件が規約外`,
  )
  for (const [l, p] of bad.slice(0, 12)) console.log(`        global.css:${l} ${p}`)
}

console.log('')
console.log('='.repeat(70))
if (failures > 0) {
  console.log(`結果: ${failures} 件の不合格`)
  process.exit(1)
}
console.log('結果: すべて合格')
