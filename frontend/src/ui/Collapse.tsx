/**
 * 開閉に合わせて高さが伸び縮みするラッパー。
 *
 * 条件付きで出し入れする UI（介入モードのスライダー、読み込み中のバナー、
 * 破壊的操作の確認）が瞬間的に現れると、その分だけ下の内容が飛ぶ。
 * どこに何が挿さったのかを目で追えるよう、高さをアニメーションさせる。
 *
 * 高さ auto は grid-template-rows: 0fr → 1fr で伸ばしている（JS で scrollHeight を
 * 測る必要がなく、中身が変わっても測り直しが要らない）。
 *
 * ★ 親が gap を持つ縦積みの中で使うときは、閉じていても gap が 1 つ分残る。
 *   gap プロパティに親の gap を渡すと負のマージンで打ち消す。
 *   先頭の子として使うと打ち消す相手がいないので、上に食い込む点に注意。
 */

import type { ReactNode } from 'react'

export interface CollapseProps {
  open: boolean
  children: ReactNode
  /** 親の gap（px）。閉じたときにこの分を詰める。Card 内なら 10、panel-body 直下なら 12 */
  gap?: number
  className?: string
}

export function Collapse({ open, children, gap = 10, className = '' }: CollapseProps) {
  return (
    <div
      className={`m3-collapse ${className}`.trim()}
      data-open={open ? 'true' : 'false'}
      style={{ ['--m3-collapse-gap' as string]: `${gap}px` }}
      // 閉じている間も DOM には残るので、読み上げとタブ移動から外しておく
      inert={!open}
    >
      <div className="m3-collapse-inner">{children}</div>
    </div>
  )
}
