/** 学習タブのヒヤリハットの難易度ゲージ（panel/CurriculumGauge.tsx）の描画テスト。 */

import assert from 'node:assert/strict'
import { register } from 'node:module'
import { describe, test } from 'node:test'

import { hooksUrl } from '../../__tests__/support/tsxHooks.ts'

register(hooksUrl)

const { createElement } = await import('react')
const { renderToStaticMarkup } = await import('react-dom/server')
const { CurriculumGauge } = await import('../CurriculumGauge.tsx')

function render(level: number | undefined, triggered: number | undefined, avoidedRate: number | null | undefined): string {
  return renderToStaticMarkup(createElement(CurriculumGauge, { level, triggered, avoidedRate }))
}

describe('CurriculumGauge', () => {
  test('0.65 でゲージの伸びが 0.65 になる（幅ではなく scaleX の変数で渡す）', () => {
    const html = render(0.65, 12, 0.8)
    assert.ok(html.includes('--m3-bar-value:0.65'), html)
    assert.ok(html.includes('Curriculum Level 65%'), html)
    assert.ok(!/width:\s*65%/.test(html), 'レイアウトを起こす width で伸ばさない（CLAUDE.md の遷移の規約）')
  })

  test('発生件数 0 のときは NaN% を出さず — にする', () => {
    const html = render(0.3, 0, null)
    assert.ok(!html.includes('NaN'), html)
    assert.match(html, /data-testid="incidents-avoided">—</)
    assert.ok(html.includes('0 回'), html)
  })

  test('件数があっても回避率が未確定（null）なら —', () => {
    assert.match(render(0.3, 3, null), /data-testid="incidents-avoided">—</)
  })

  test('回避率は小数 1 桁で出す', () => {
    assert.match(render(0.5, 7, 0.857), /data-testid="incidents-avoided">85\.7%</)
  })

  test('値が届いていない（古いサーバー）ときは例外なく描けて、ゲージは 0', () => {
    const html = render(undefined, undefined, undefined)
    assert.ok(html.includes('Curriculum Level —'), html)
    assert.ok(html.includes('--m3-bar-value:0'), html)
    assert.ok(!html.includes('NaN'), html)
  })

  test('範囲外の値は 0〜1 に収める', () => {
    assert.ok(render(1.4, 1, 2).includes('--m3-bar-value:1'))
    assert.ok(render(-0.2, 1, -1).includes('--m3-bar-value:0'))
  })
})
