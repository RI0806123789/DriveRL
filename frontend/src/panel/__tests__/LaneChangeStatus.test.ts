/** 車線変更の状態・根拠・実介入・記録量を画面から区別できるかを検査する。 */

import assert from 'node:assert/strict'
import { register } from 'node:module'
import { describe, test } from 'node:test'

import { hooksUrl } from '../../__tests__/support/tsxHooks.ts'
import type { LaneChangeDecision } from '../../types/protocol.ts'
import { assistSummary, laneChangeSummary, paneAssist } from '../../scene/assistLabels.ts'

register(hooksUrl)

const { createElement } = await import('react')
const { renderToStaticMarkup } = await import('react-dom/server')
const { LaneChangeStatus } = await import('../LaneChangeStatus.tsx')

function decision(patch: Partial<LaneChangeDecision> = {}): LaneChangeDecision {
  return {
    state: 'waiting',
    reason: 'gap',
    source: 'camera',
    direction: -1,
    imposed: false,
    minGap: 8.25,
    rearDecel: 1.25,
    lateralAccel: 0.2,
    lateralJerk: -0.3,
    completed: 2,
    ...patch,
  }
}

function status(value: LaneChangeDecision | undefined): string {
  return renderToStaticMarkup(createElement(LaneChangeStatus, { decision: value }))
}

describe('車線変更の判断表示', () => {
  test('変更先・理由・カメラ根拠・計測量を表示し、判定と実介入を区別する', () => {
    const observed = status(decision())
    for (const text of ['右へ変更待ち', '前後車間の空きを待つ', 'カメラ判断', '判定のみ', '最小車間 8.3m',
      '後続減速度 1.3m/s²', '横加速度 0.2m/s²', '横ジャーク -0.3m/s³', '回避完了 2回', 'data-imposed="false"']) {
      assert.ok(observed.includes(text), observed)
    }
    const imposed = status(decision({ imposed: true, source: 'background' }))
    assert.ok(imposed.includes('操作介入あり') && imposed.includes('背景交通の判断'), imposed)
    assert.ok(imposed.includes('data-imposed="true"') && !imposed.includes('判定のみ'), imposed)
  })

  test('変更・通過・復帰・完了へ表示が変わり、方向なしでは左右を表示しない', () => {
    for (const [state, text] of [
      ['changing', '変更中'], ['passing', '障害物を通過中'], ['returning', '復帰中'],
      ['complete', '回避完了'], ['rerouting', '迂回経路へ変更'],
    ] as const) {
      const html = status(decision({ state, reason: '', direction: 0 }))
      assert.ok(html.includes(text), html)
      assert.ok(!html.includes('右へ') && !html.includes('左へ'), html)
    }
    assert.ok(status(decision({ state: 'returning', direction: 1 })).includes('左へ復帰中'))
    assert.equal(status(undefined), '')
  })

  test('高速後続車・検出不足・変更禁止・規制不明・交通待ちを理由として区別する', () => {
    for (const [reason, text] of [
      ['gap', '前後車間の空きを待つ'], ['visibility', 'カメラ情報・見通しを確認'],
      ['regulation', '車線変更禁止'], ['unknown_regulation', '車線変更の規制が不明'],
      ['traffic', '信号待ち・渋滞に追従'],
    ] as const) {
      const html = status(decision({ reason }))
      assert.ok(html.includes(text) && html.includes('lane-change-status--warn'), html)
      assert.ok(html.includes('判定のみ'), html)
    }
  })

  test('全車線閉鎖と迂回不能を停止理由として区別する', () => {
    for (const [reason, text] of [['blocked', '全車線が塞がっている'], ['no_detour', '迂回経路がない']] as const) {
      const html = status(decision({ state: 'blocked', reason, direction: 0, source: 'road_event' }))
      assert.ok(html.includes(`通過できない（${text}）`) && html.includes('lane-change-status--stop'), html)
      assert.ok(html.includes('道路規制') && !html.includes('カメラ判断'), html)
    }
  })

  test('未知・非有限の計測量を画面へ出さず、重なった車間の負値は保持する', () => {
    const html = status(decision({ minGap: null, rearDecel: Number.NaN, lateralAccel: Infinity, lateralJerk: -Infinity, completed: Number.NaN }))
    assert.ok(!html.includes('NaN') && !html.includes('Infinity'), html)
    assert.ok(!html.includes('最小車間') && !html.includes('後続減速度') && !html.includes('横加速度') && !html.includes('横ジャーク'), html)
    assert.ok(status(decision({ minGap: -0.5 })).includes('最小車間 -0.5m'))
    assert.equal(laneChangeSummary(decision({ state: 'unknown' as LaneChangeDecision['state'] })), null)
    assert.equal(laneChangeSummary(decision({ reason: 'unknown' as LaneChangeDecision['reason'] })), null)
  })
})

describe('車線変更の介入表示', () => {
  test('変更・待機・復帰・閉鎖を日本語で表示し、前方ペインに根拠を示す', () => {
    for (const kind of ['lane_change', 'lane_wait', 'lane_return', 'road_closed'] as const) {
      assert.notEqual(assistSummary(kind), null, kind)
      assert.notEqual(paneAssist(kind, 'front', 1), null, kind)
    }
  })

  test('後方の確認と、ウインカー側の変更・復帰を表示する', () => {
    for (const kind of ['lane_change', 'lane_wait', 'lane_return'] as const) {
      assert.notEqual(paneAssist(kind, 'rear', 1), null, kind)
    }
    for (const kind of ['lane_change', 'lane_return'] as const) {
      assert.notEqual(paneAssist(kind, 'right', 1), null, kind)
      assert.equal(paneAssist(kind, 'left', 1), null, kind)
      assert.notEqual(paneAssist(kind, 'left', -1), null, kind)
    }
    assert.notEqual(paneAssist('lane_wait', 'left', 0), null)
    assert.notEqual(paneAssist('lane_wait', 'right', 0), null)
    assert.equal(paneAssist('road_closed', 'rear', 0), null)
  })
})
