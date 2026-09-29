/** 学習中の車のうちエキスパート（経路追従）が運転した割合のチップ。 */

import { assistRateView } from '../store/assistRate'
import { Chip } from '../ui/Chip'

export function AssistRateChip({ rate }: { rate: number | undefined }) {
  const view = assistRateView(rate)
  return (
    <Chip tone={view.tone} title="直近 10 秒の学習中の車のステップのうち、経路追従が代わりに運転した割合">
      {view.text}
    </Chip>
  )
}
