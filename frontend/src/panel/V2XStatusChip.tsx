/** 追跡中の車が V2X でメッセージを受け取っている相手のチップ。相手がいなければ何も出さない。 */

import { v2xStatusText } from '../scene/v2xLinks'
import { Chip } from '../ui/Chip'

export function V2XStatusChip({ links }: { links: readonly number[] | undefined }) {
  const text = v2xStatusText(links)
  if (text === null) return null
  return (
    <Chip small title="車車間通信（30m 以内の近い 2 台）で、車速・右左折の意図・危険・交差点への近さを受け取っています">
      {text}
    </Chip>
  )
}
