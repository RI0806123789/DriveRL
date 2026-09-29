/** シミュレーションタブの V2X のチップ（panel/V2XStatusChip.tsx）の描画テスト。 */

import assert from 'node:assert/strict'
import { register } from 'node:module'
import { describe, test } from 'node:test'

import { hooksUrl } from '../../__tests__/support/tsxHooks.ts'

register(hooksUrl)

const { createElement } = await import('react')
const { renderToStaticMarkup } = await import('react-dom/server')
const { V2XStatusChip } = await import('../V2XStatusChip.tsx')

function render(links: number[] | undefined): string {
  return renderToStaticMarkup(createElement(V2XStatusChip, { links }))
}

describe('V2XStatusChip', () => {
  test('相手が 1 台なら「V2X: 車両#2とリンク中」のチップを出す', () => {
    const html = render([2])
    assert.match(html, /class="m3-chip[^"]*"/)
    assert.ok(html.includes('V2X: 車両#2とリンク中'), html)
  })

  test('相手が 2 台なら近い順に並べる', () => {
    assert.ok(render([5, 1]).includes('V2X: 車両#5・#1とリンク中'))
  })

  test('通信が切れたら（空・未定義）何も描かない', () => {
    assert.equal(render([]), '')
    assert.equal(render(undefined), '')
  })

  test('つながる → 切れる → つながる と描き直しても例外にならない', () => {
    for (const links of [[3], [], [3, 4], undefined, [4]]) render(links)
  })
})
