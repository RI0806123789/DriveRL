/** 学習タブのアシスト率のチップ（panel/AssistRateChip.tsx）の描画テスト。 */

import assert from 'node:assert/strict'
import { register } from 'node:module'
import { describe, test } from 'node:test'

import { hooksUrl } from '../../__tests__/support/tsxHooks.ts'

register(hooksUrl)

const { createElement } = await import('react')
const { renderToStaticMarkup } = await import('react-dom/server')
const { AssistRateChip } = await import('../AssistRateChip.tsx')

function render(rate: number | undefined): string {
  return renderToStaticMarkup(createElement(AssistRateChip, { rate }))
}

describe('AssistRateChip', () => {
  test('0.85 を与えると「Assist Rate 85%」のチップが描かれる', () => {
    const html = render(0.85)
    assert.match(html, /class="m3-chip[^"]*"/)
    assert.ok(html.includes('Assist Rate 85%'), html)
  })

  test('0% へ変わっても例外なく描き直せる', () => {
    assert.ok(render(0.85).includes('85%'))
    const html = render(0)
    assert.ok(html.includes('Assist Rate 0%'), html)
    assert.ok(html.includes('m3-chip--tone-ok'), html)
  })

  test('値が届いていない（古いサーバー）ときは — を出す', () => {
    assert.ok(render(undefined).includes('Assist Rate —'))
  })

  test('範囲外・数でない値は 0〜100% に収める', () => {
    assert.ok(render(1.7).includes('Assist Rate 100%'))
    assert.ok(render(-0.2).includes('Assist Rate 0%'))
    assert.ok(render(Number.NaN).includes('Assist Rate —'))
  })

  test('高いうちは注意の色で出す', () => {
    assert.ok(render(0.85).includes('m3-chip--tone-warning'))
  })
})
