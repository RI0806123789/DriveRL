/** スマホ画面の車載 AI コンシェルジュ。地図の上へ重ねるので、開閉でレイアウトは動かない。 */

import { useEffect, useState } from 'react'
import type { FormEvent } from 'react'

import {
  CONCIERGE_GREETING,
  CONCIERGE_MAX_CHARS,
  DRIVE_MODE_LABELS,
  QUICK_ACTION_LABELS,
  useConcierge,
} from '../store/concierge'
import type { ConciergeAction, ConciergeEntry } from '../store/concierge'
import { useSimStore } from '../store/simStore'
import type { TaxiDriveMode } from '../types/protocol'
import { CloseIcon, SendIcon, SparkleIcon } from '../ui/Icons'

const QUICK_ACTIONS: Exclude<ConciergeAction, 'normal'>[] = ['hurry', 'comfort', 'explain']

export interface TaxiAiViewBodyProps {
  on: boolean
  /** null はまだ確かめていない */
  available: boolean | null
  /** モック接続中なら問い合わせず、使えない理由を出す */
  mock?: boolean
  pending: boolean
  entries: ConciergeEntry[]
  driveMode: TaxiDriveMode
  draft: string
  onDraft(text: string): void
  onAsk(action: ConciergeAction): void
  onSend(text: string): void
  onClose(): void
}

/** 最新の返答（無ければあいさつ）と、それに対する乗客の発言。 */
export function latestExchange(entries: ConciergeEntry[]): { said: string | null; reply: ConciergeEntry | null } {
  let reply: ConciergeEntry | null = null
  let said: string | null = null
  for (let i = entries.length - 1; i >= 0; i--) {
    const e = entries[i]
    if (e.role === 'user') {
      said = e.text
      break
    }
    if (!reply) reply = e
  }
  return { said, reply }
}

export function TaxiAiViewBody(props: TaxiAiViewBodyProps) {
  const { on, available, mock = false, pending, entries, driveMode, draft } = props
  const usable = !mock && available === true
  const { said, reply } = latestExchange(entries)

  let cardText = CONCIERGE_GREETING
  let cardRole: ConciergeEntry['role'] = 'ai'
  if (mock) {
    cardText = 'モック接続中は AI コンシェルジュを使えません。実際のバックエンドに繋いでください'
    cardRole = 'error'
  } else if (available === false) {
    cardText = 'GEMINI_API_KEY が設定されていないため、AI コンシェルジュは使えません（.env.example を参照）'
    cardRole = 'error'
  } else if (pending) {
    cardText = '考えています…'
  } else if (reply) {
    cardText = reply.text
    cardRole = reply.role
  }

  const submit = (e: FormEvent) => {
    e.preventDefault()
    if (!usable || pending || !draft.trim()) return
    props.onSend(draft)
  }

  return (
    <div className="taxi-ai" data-on={on ? 'true' : 'false'} aria-hidden={!on}>
      <div className="taxi-ai-inner" inert={!on}>
        <div className="taxi-ai-head">
          <SparkleIcon size={14} />
          <span className="taxi-ai-title">AI コンシェルジュ</span>
          <span className="taxi-ai-mode" data-mode={driveMode}>
            {DRIVE_MODE_LABELS[driveMode]}
          </span>
          <button type="button" className="taxi-ai-close" aria-label="AI コンシェルジュを閉じる" onClick={props.onClose}>
            <CloseIcon size={14} />
          </button>
        </div>

        {said && !pending && <div className="taxi-ai-said">{said}</div>}
        <div
          className="taxi-ai-card"
          data-role={cardRole}
          data-pending={pending ? 'true' : 'false'}
          aria-live="polite"
          key={reply?.id ?? (pending ? 'pending' : 'greeting')}
        >
          {cardText}
        </div>

        <div className="taxi-ai-chips">
          {QUICK_ACTIONS.map((action) => (
            <button
              key={action}
              type="button"
              className="taxi-ai-chip"
              data-active={action === driveMode ? 'true' : 'false'}
              disabled={!usable || pending}
              onClick={() => props.onAsk(action === driveMode ? 'normal' : action)}
            >
              {QUICK_ACTION_LABELS[action]}
            </button>
          ))}
        </div>

        <form className="taxi-ai-form" onSubmit={submit}>
          <input
            className="taxi-ai-input"
            type="text"
            value={draft}
            maxLength={CONCIERGE_MAX_CHARS}
            placeholder={usable ? 'ご要望をどうぞ（例: 揺れを抑えて）' : '使えません'}
            disabled={!usable}
            onChange={(e) => props.onDraft(e.target.value)}
          />
          <button
            type="submit"
            className="taxi-ai-send"
            aria-label="送信"
            disabled={!usable || pending || !draft.trim()}
          >
            <SendIcon size={15} />
          </button>
        </form>
      </div>
    </div>
  )
}

export interface TaxiAiViewProps {
  on: boolean
  driveMode: TaxiDriveMode
  onClose(): void
}

export function TaxiAiView({ on, driveMode, onClose }: TaxiAiViewProps) {
  const available = useConcierge((s) => s.available)
  const pending = useConcierge((s) => s.pending)
  const entries = useConcierge((s) => s.entries)
  const ask = useConcierge((s) => s.ask)
  const checkAvailability = useConcierge((s) => s.checkAvailability)
  const usingMock = useSimStore((s) => s.usingMock)
  const rideId = useSimStore((s) => s.taxi.rideId)
  const [draft, setDraft] = useState('')

  // キーはサーバーの起動時に読むので、開いたときに確かめれば足りる。モックでは本物のサーバーへ問い合わせない
  useEffect(() => {
    if (on && !usingMock) void checkAvailability()
  }, [on, usingMock, checkAvailability])

  return (
    <TaxiAiViewBody
      on={on}
      available={available}
      mock={usingMock}
      pending={pending}
      entries={entries}
      driveMode={driveMode}
      draft={draft}
      onDraft={setDraft}
      onAsk={(action) => void ask({ action, rideId })}
      onSend={(text) => {
        setDraft('')
        void ask({ message: text, rideId })
      }}
      onClose={onClose}
    />
  )
}
