/** 交差点判断の理由・待機・根拠・介入を、利用者が表示から区別できるかを検査する。 */

import assert from 'node:assert/strict'
import { register } from 'node:module'
import { describe, test } from 'node:test'

import { hooksUrl } from '../../__tests__/support/tsxHooks.ts'
import type { AssistKind, IntersectionDecision } from '../../types/protocol.ts'
import { assistSummary, intersectionSummary, paneAssist } from '../../scene/assistLabels.ts'

register(hooksUrl)

const { createElement } = await import('react')
const { renderToStaticMarkup } = await import('react-dom/server')
const { IntersectionStatus } = await import('../IntersectionStatus.tsx')

function decision(patch: Partial<IntersectionDecision> = {}): IntersectionDecision {
  return {
    nodeId: 12,
    state: 'waiting',
    reason: 'oncoming',
    source: 'camera',
    waitingSeconds: 3.25,
    gapSeconds: 1.4,
    imposed: false,
    ...patch,
  }
}

function status(value: IntersectionDecision | undefined): string {
  return renderToStaticMarkup(createElement(IntersectionStatus, { decision: value }))
}

describe('交差点の判断表示', () => {
  test('対向車を待つ理由・待機秒数・カメラの根拠を表示し、判定のみと介入を区別する', () => {
    const observed = status(decision())
    assert.ok(observed.includes('進入待ち（対向車に譲る）'), observed)
    assert.ok(observed.includes('待機 3.3秒'), observed)
    assert.ok(observed.includes('間隔 1.4秒'), observed)
    assert.ok(observed.includes('カメラ判断'), observed)
    assert.ok(observed.includes('判定のみ'), observed)
    assert.ok(observed.includes('data-imposed="false"'), observed)
    const imposed = status(decision({ imposed: true }))
    assert.ok(imposed.includes('操作介入あり'), imposed)
    assert.ok(!imposed.includes('判定のみ'), imposed)
  })

  test('背景交通の判断をカメラ判断と表示しない', () => {
    const html = status(decision({ source: 'background', imposed: true }))
    assert.ok(html.includes('背景交通の判断'), html)
    assert.ok(!html.includes('カメラ判断'), html)
    assert.ok(html.includes('data-source="background"'), html)
  })

  test('未検出・古いカメラ情報は進入可能に見せず、確認待ちを表示する', () => {
    const html = status(decision({ reason: 'visibility', gapSeconds: null }))
    assert.ok(html.includes('見通し・カメラ情報を確認'), html)
    assert.ok(html.includes('intersection-status--warn'), html)
    assert.ok(!html.includes('進入可能'), html)
    assert.ok(!html.includes('間隔'), html)
  })

  test('横断者・出口の詰まり・一時停止・優先判断を理由として区別する', () => {
    for (const [reason, text] of [
      ['pedestrian', '横断者に譲る'],
      ['exit_blocked', '出口の空きを待つ'],
      ['stop', '一時停止'],
      ['priority', '優先される交通に譲る'],
    ] as const) {
      const html = status(decision({ reason }))
      assert.ok(html.includes(text), html)
      assert.ok(html.includes('intersection-status--stop'), html)
    }
  })

  test('接近・安全な進入・通過中へ表示が変わり、対象なしでは消える', () => {
    for (const [state, text] of [['approaching', '接近中'], ['clear', '進入可能'], ['committed', '通過中']] as const) {
      const html = status(decision({ state, reason: '', waitingSeconds: 0, gapSeconds: null }))
      assert.ok(html.includes(text), html)
      assert.ok(!html.includes('対向車に譲る'), html)
      assert.ok(!html.includes('intersection-status--stop'), html)
    }
    assert.equal(status(undefined), '')
  })

  test('不正な待機秒数・非有限の間隔を画面へ出さない', () => {
    const html = status(decision({ waitingSeconds: Number.NaN, gapSeconds: Number.POSITIVE_INFINITY }))
    assert.ok(html.includes('待機 0.0秒'), html)
    assert.ok(!html.includes('NaN') && !html.includes('Infinity'), html)
    assert.equal(intersectionSummary(decision({ waitingSeconds: -2 }))?.detail.includes('待機 0.0秒'), true)
  })
})

describe('交差点の介入表示', () => {
  test('各介入に日本語の概要と、根拠を見る前方ペインを持つ', () => {
    for (const kind of ['intersection_yield', 'intersection_stop', 'intersection_exit', 'intersection_visibility'] as const) {
      assert.ok(assistSummary(kind)?.text.includes('交差点'), kind)
      assert.notEqual(paneAssist(kind, 'front', 1), null, kind)
      assert.equal(paneAssist(kind, 'rear', 1), null, kind)
    }
  })

  test('交錯交通・見通しは側方にも表示し、一時停止・出口は前方だけに表示する', () => {
    for (const pane of ['left', 'right'] as const) {
      assert.notEqual(paneAssist('intersection_yield', pane, 0), null)
      assert.notEqual(paneAssist('intersection_visibility', pane, 0), null)
      assert.equal(paneAssist('intersection_stop', pane, 0), null)
      assert.equal(paneAssist('intersection_exit', pane, 0), null)
    }
    assert.equal(assistSummary('' as AssistKind), null)
  })
})
