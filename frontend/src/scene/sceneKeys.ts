/** 実用モードの 3D 操作（WASD・乗降・緊急停止）を、画面の部品の操作と取り合わないための判定。DOM を持たない純粋な関数。 */

/** 判定に使う要素の性質だけを持つ形（node のテストで偽の要素を渡せるように） */
export interface ElementLike {
  tagName?: string
  isContentEditable?: boolean
  getAttribute?(name: string): string | null
  hasAttribute?(name: string): boolean
  parentElement?: ElementLike | null
}

export interface KeyEventLike {
  target: EventTarget | ElementLike | null
  defaultPrevented: boolean
  ctrlKey: boolean
  metaKey: boolean
  altKey: boolean
  isComposing: boolean
}

/** キーを受けると自分で何かをする要素（Enter / Space で押す・文字を打つ・値を動かす） */
const CONTROL_TAGS = new Set(['INPUT', 'TEXTAREA', 'SELECT', 'BUTTON', 'SUMMARY', 'OPTION'])

const CONTROL_ROLES = new Set([
  'button', 'link', 'checkbox', 'switch', 'radio', 'menuitem', 'menuitemcheckbox', 'menuitemradio',
  'tab', 'slider', 'option', 'combobox', 'listbox', 'textbox', 'searchbox', 'spinbutton', 'treeitem',
])

/** 親をたどる上限（循環した偽の要素でも止まるように） */
const MAX_DEPTH = 64

/** キーの対象が、画面の操作部品かその中の要素か。 */
export function isUiControlTarget(target: KeyEventLike['target']): boolean {
  let el = target as ElementLike | null
  for (let depth = 0; el && depth < MAX_DEPTH; depth++) {
    if (el.isContentEditable) return true
    const tag = typeof el.tagName === 'string' ? el.tagName.toUpperCase() : ''
    if (CONTROL_TAGS.has(tag)) return true
    if (tag === 'A' && el.hasAttribute?.('href')) return true
    const role = el.getAttribute?.('role')
    if (role && role.split(/\s+/).some((r) => CONTROL_ROLES.has(r.toLowerCase()))) return true
    el = el.parentElement ?? null
  }
  return false
}

/** この keydown を 3D の操作として扱ってよいか。部品の操作・ほかの部品が処理した後・修飾キーつき・変換中は扱わない。 */
export function isSceneKeyEvent(e: KeyEventLike): boolean {
  if (e.defaultPrevented) return false
  if (e.ctrlKey || e.metaKey || e.altKey || e.isComposing) return false
  return !isUiControlTarget(e.target)
}
