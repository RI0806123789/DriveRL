/** 実用モードの 3D 操作のキー（scene/sceneKeys.ts）が、画面の部品の Enter / Space を横取りしないかの単体テスト。 */

import assert from 'node:assert/strict'
import { describe, test } from 'node:test'

import { isSceneKeyEvent, isUiControlTarget, type ElementLike, type KeyEventLike } from '../scene/sceneKeys.ts'

function el(tagName: string, attrs: Record<string, string> = {}, parent: ElementLike | null = null, editable = false): ElementLike {
  return {
    tagName,
    isContentEditable: editable,
    getAttribute: (name) => attrs[name] ?? null,
    hasAttribute: (name) => name in attrs,
    parentElement: parent,
  }
}

function key(target: KeyEventLike['target'], extra: Partial<KeyEventLike> = {}): KeyEventLike {
  return { target, defaultPrevented: false, ctrlKey: false, metaKey: false, altKey: false, isComposing: false, ...extra }
}

const body = el('BODY')
const panel = el('DIV', { class: 'panel' }, body)

describe('3D の操作として扱うキー', () => {
  test('画面の部品（とその中の要素）では扱わない', () => {
    const cases: [string, ElementLike][] = [
      ['button', el('BUTTON', {}, panel)],
      ['button の中のアイコン', el('svg', {}, el('SPAN', {}, el('BUTTON', {}, panel)))],
      ['リンク', el('A', { href: '#' }, panel)],
      ['role=button', el('DIV', { role: 'button', tabindex: '0' }, panel)],
      ['role=switch の中', el('SPAN', {}, el('DIV', { role: 'switch' }, panel))],
      ['role=tab', el('DIV', { role: 'tab' }, panel)],
      ['role=slider', el('DIV', { role: 'slider' }, panel)],
      ['contenteditable の子', el('SPAN', {}, el('DIV', {}, panel, true), true)],
      ['input', el('INPUT', { type: 'range' }, panel)],
      ['textarea', el('TEXTAREA', {}, panel)],
      ['select', el('SELECT', {}, panel)],
      ['summary', el('SUMMARY', {}, panel)],
    ]
    for (const [label, target] of cases) {
      assert.equal(isUiControlTarget(target), true, label)
      assert.equal(isSceneKeyEvent(key(target)), false, label)
    }
  })

  test('3D の画面・何もない所・ページ全体では扱う（乗降・緊急停止・WASD を保つ）', () => {
    const canvas = el('CANVAS', {}, el('DIV', { class: 'viewport' }, body))
    const cases: [string, KeyEventLike['target']][] = [
      ['body', body],
      ['canvas', canvas],
      ['パネルの文字', el('P', {}, panel)],
      ['href の無い a', el('A', {}, panel)],
      ['role=presentation', el('DIV', { role: 'presentation' }, panel)],
      ['window（tagName が無い）', {} as ElementLike],
      ['null', null],
    ]
    for (const [label, target] of cases) {
      assert.equal(isSceneKeyEvent(key(target)), true, label)
    }
  })

  test('ほかの部品が処理した後・修飾キーつき・変換中は扱わない', () => {
    assert.equal(isSceneKeyEvent(key(body, { defaultPrevented: true })), false)
    assert.equal(isSceneKeyEvent(key(body, { ctrlKey: true })), false)
    assert.equal(isSceneKeyEvent(key(body, { metaKey: true })), false)
    assert.equal(isSceneKeyEvent(key(body, { altKey: true })), false)
    assert.equal(isSceneKeyEvent(key(body, { isComposing: true })), false)
  })

  test('親が循環した偽の要素でも止まる', () => {
    const a = el('DIV') as { parentElement?: ElementLike | null } & ElementLike
    const b = el('DIV', {}, a)
    a.parentElement = b
    assert.equal(isUiControlTarget(a), false)
  })
})
