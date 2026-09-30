/** 実用モードの AI コンシェルジュ（panel/TaxiAiView.tsx）と、応答の読み方（store/concierge.ts）の描画テスト。 */

import assert from 'node:assert/strict'
import { register } from 'node:module'
import { describe, test } from 'node:test'

import { hooksUrl } from '../../__tests__/support/tsxHooks.ts'

register(hooksUrl)

const { createElement } = await import('react')
const { renderToStaticMarkup } = await import('react-dom/server')
const { TaxiAiViewBody, latestExchange } = await import('../TaxiAiView.tsx')
const { CONCIERGE_GREETING, entryFromResponse } = await import('../../store/concierge.ts')

type Props = Parameters<typeof TaxiAiViewBody>[0]

function view(patch: Partial<Props> = {}): string {
  const props: Props = {
    on: true,
    available: true,
    pending: false,
    entries: [],
    driveMode: 'normal',
    draft: '',
    onDraft: () => {},
    onAsk: () => {},
    onSend: () => {},
    onClose: () => {},
    ...patch,
  }
  return renderToStaticMarkup(createElement(TaxiAiViewBody, props))
}

function count(html: string, needle: string): number {
  return html.split(needle).length - 1
}

describe('TaxiAiViewBody', () => {
  test('開いているときは data-on が true で、閉じると inert になる', () => {
    assert.ok(view().includes('class="taxi-ai" data-on="true"'))
    const closed = view({ on: false })
    assert.ok(closed.includes('data-on="false"'), closed)
    assert.ok(closed.includes('inert'), closed)
  })

  test('返答がまだ無ければあいさつを出し、3 つのチップを押せる', () => {
    const html = view()
    assert.ok(html.includes(CONCIERGE_GREETING), html)
    assert.equal(count(html, 'class="taxi-ai-chip"'), 3)
    assert.ok(html.includes('少し急いで') && html.includes('快適重視') && html.includes('停車理由'), html)
    assert.equal(count(html, 'disabled=""'), 1, '送信ボタンだけが押せない（入力が空）')
  })

  test('キーが無いときは理由を出し、チップも入力も押せない', () => {
    const html = view({ available: false })
    assert.ok(html.includes('GEMINI_API_KEY'), html)
    assert.ok(html.includes('data-role="error"'), html)
    assert.equal(count(html, 'disabled=""'), 5, 'チップ 3 つ + 入力 + 送信')
  })

  test('いまの走り方のチップが点き、走り方の名前が見出しに出る', () => {
    const html = view({ driveMode: 'hurry' })
    assert.ok(html.includes('data-mode="hurry"'), html)
    assert.ok(html.includes('>少し急いで</span>'), html)
    assert.equal(count(html, 'data-active="true"'), 1)
  })

  test('問い合わせ中は「考えています」を出し、チップと送信を止める', () => {
    const html = view({ pending: true, draft: '急いで' })
    assert.ok(html.includes('考えています'), html)
    assert.ok(html.includes('data-pending="true"'), html)
    assert.equal(count(html, 'disabled=""'), 4, 'チップ 3 つ + 送信（入力は打てる）')
  })

  test('最新の返答と、それに対する乗客の発言を出す', () => {
    const html = view({
      entries: [
        { id: 1, role: 'user', text: '前の質問' },
        { id: 2, role: 'ai', text: '前の返答' },
        { id: 3, role: 'user', text: 'なぜ止まっているの' },
        { id: 4, role: 'ai', text: '前方信号待ちのため停車中です。' },
      ],
    })
    assert.ok(html.includes('なぜ止まっているの'), html)
    assert.ok(html.includes('前方信号待ちのため停車中です。'), html)
    assert.ok(!html.includes('前の返答'), html)
  })
})

describe('latestExchange', () => {
  test('返答がまだ来ていなければ、発言だけを返す', () => {
    const got = latestExchange([{ id: 1, role: 'user', text: '急いで' }])
    assert.equal(got.said, '急いで')
    assert.equal(got.reply, null)
  })
})

describe('entryFromResponse', () => {
  test('ok の返答は AI の発言にする', () => {
    assert.deepEqual(entryFromResponse(200, { ok: true, reply: 'はい' }), { role: 'ai', text: 'はい' })
  })

  test('失敗はサーバーの理由をそのまま出し、理由が無ければ HTTP の番号を出す', () => {
    assert.deepEqual(entryFromResponse(503, { ok: false, error: 'キーがありません' }), {
      role: 'error',
      text: 'キーがありません',
    })
    assert.equal(entryFromResponse(500, null).role, 'error')
    assert.ok(entryFromResponse(500, null).text.includes('500'))
  })
})
