/** package.json の verify:* を 1 本ずつ子プロセスで回し、終了コードで合否を決める。 */

import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { readdirSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { test } from 'node:test'
import { fileURLToPath } from 'node:url'

const FRONTEND_DIR = join(dirname(fileURLToPath(import.meta.url)), '..', '..')
const TIMEOUT_MS = 600_000

const pkg = JSON.parse(readFileSync(join(FRONTEND_DIR, 'package.json'), 'utf8')) as {
  scripts: Record<string, string>
}
const entries = Object.entries(pkg.scripts).filter(([name]) => name.startsWith('verify:'))

function scriptPath(command: string): string | null {
  return /^node (scripts\/[\w-]+\.ts)$/.exec(command)?.[1] ?? null
}

test('package.json に verify:* が 1 本以上ある', () => {
  assert.ok(entries.length > 0)
})

test('npm run verify は verify:* を漏れなく 1 回ずつ呼ぶ', () => {
  const chained = pkg.scripts.verify.split('&&').map((s) => s.trim().replace(/^npm run /, ''))
  assert.deepEqual([...chained].sort(), entries.map(([name]) => name).sort())
})

test('scripts/verify-*.ts はすべて verify:* から呼ばれている', () => {
  const registered = new Set(entries.map(([, command]) => scriptPath(command)))
  const files = readdirSync(join(FRONTEND_DIR, 'scripts')).filter((f) => /^verify-.*\.ts$/.test(f))
  assert.deepEqual(
    files.filter((f) => !registered.has(`scripts/${f}`)),
    [],
    'package.json の scripts に登録されていない検証スクリプト',
  )
})

for (const [name, command] of entries) {
  test(name, () => {
    const path = scriptPath(command)
    assert.ok(path, `${name} は "node scripts/verify-xxx.ts" の形にすること（${command}）`)
    const proc = spawnSync(process.execPath, [path], {
      cwd: FRONTEND_DIR,
      encoding: 'utf8',
      timeout: TIMEOUT_MS,
    })
    if (proc.status === 0) return
    const lines = `${proc.stdout ?? ''}${proc.stderr ?? ''}`.split(/\r?\n/)
    const failed = lines.filter((line) => line.includes('[NG'))
    assert.fail(
      [
        `${command} が終了コード ${proc.status} で終わった`,
        ...failed,
        '--- 末尾 ---',
        ...lines.slice(-30),
      ].join('\n'),
    )
  })
}
