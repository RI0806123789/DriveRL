/** 意図の色付きバッジ（panel/OptionBadge.tsx）と、学習タブの意図の割合（panel/OptionShares.tsx）の描画テスト。 */

import assert from 'node:assert/strict'
import { register } from 'node:module'
import { describe, test } from 'node:test'

import { hooksUrl } from '../../__tests__/support/tsxHooks.ts'

register(hooksUrl)

const { createElement } = await import('react')
const { renderToStaticMarkup } = await import('react-dom/server')
const { OptionBadge } = await import('../OptionBadge.tsx')
const { OptionShares } = await import('../OptionShares.tsx')

function badge(option: unknown): string {
  return renderToStaticMarkup(createElement(OptionBadge, { option }))
}

describe('OptionBadge', () => {
  test('STOP を渡すと赤のバッジ「STOP」を描く', () => {
    const html = badge('STOP')
    assert.ok(html.includes('option-badge--stop'), html)
    assert.ok(html.includes('data-option="STOP"'), html)
    assert.ok(html.includes('>STOP</span>'), html)
    assert.ok(!html.includes('option-badge--cruise'), html)
  })

  test('CRUISE に替えると青のバッジへ切り替わる', () => {
    const html = badge('CRUISE')
    assert.ok(html.includes('option-badge--cruise'), html)
    assert.ok(html.includes('>CRUISE</span>'), html)
    assert.ok(!html.includes('option-badge--stop'), html)
  })

  test('FOLLOW は緑・YIELD は黄のクラス', () => {
    assert.ok(badge('FOLLOW').includes('option-badge--follow'))
    assert.ok(badge('YIELD').includes('option-badge--yield'))
  })

  test('意図が無い・知らない名前なら何も描かない', () => {
    assert.equal(badge(undefined), '')
    assert.equal(badge(''), '')
    assert.equal(badge('TURBO'), '')
    assert.equal(badge(2), '')
  })
})

describe('OptionShares', () => {
  function shares(value: unknown): string {
    return renderToStaticMarkup(createElement(OptionShares, { shares: value }))
  }

  test('4 つの意図の割合を CRUISE → STOP の順に、バーは scaleX の変数で伸ばす', () => {
    const html = shares([0.5, 0.25, 0.15, 0.1])
    const order = ['CRUISE', 'FOLLOW', 'YIELD', 'STOP'].map((o) => html.indexOf(`option-shares-row" data-option="${o}"`))
    assert.ok(order.every((i) => i >= 0), html)
    assert.deepEqual([...order].sort((a, b) => a - b), order)
    assert.ok(html.includes('--m3-bar-value:0.5'), html)
    assert.ok(html.includes('>50%<') && html.includes('>10%<'), html)
    assert.ok(!/width:\s*\d/.test(html), 'レイアウトを起こす width で伸ばさない')
  })

  test('届いていない・形が違うときは案内だけ出す（NaN を出さない）', () => {
    for (const value of [undefined, [], [1, 2], [0, 0, 0, 0], [0.5, Number.NaN, 0.2, 0.3]]) {
      const html = shares(value)
      assert.ok(html.includes('方策が運転したステップがまだありません'), html)
      assert.ok(!html.includes('NaN'), html)
    }
  })
})
