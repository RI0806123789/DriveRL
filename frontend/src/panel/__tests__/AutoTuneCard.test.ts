/** 学習タブの「学習の自動化」（panel/AutoTuneCard.tsx・ui/Slider.tsx の auto・store/autotune.ts）の描画テスト。 */

import assert from 'node:assert/strict'
import { register } from 'node:module'
import { describe, test } from 'node:test'

import { hooksUrl } from '../../__tests__/support/tsxHooks.ts'

register(hooksUrl)

const { createElement } = await import('react')
const { renderToStaticMarkup } = await import('react-dom/server')
const { AutoTuneCard } = await import('../AutoTuneCard.tsx')
const { Slider } = await import('../../ui/Slider.tsx')
const store = await import('../../store/autotune.ts')
const { MOCK_TUNED_KEYS } = await import('../../store/mock/autotune.ts')

type Msg = import('../../types/protocol.ts').AutotuneMessage
type Ctx = import('../../store/autotune.ts').AutotuneContext

const READY: Ctx = { connected: true, practical: false, suspended: false, mapLoaded: true }

function message(patch: Partial<Msg> = {}): Msg {
  return {
    type: 'autotune',
    available: true,
    running: false,
    phase: 'idle',
    studyName: null,
    trial: null,
    trialProgress: 0,
    trialSteps: 1536,
    finishedTrials: 0,
    priorTrials: 0,
    tunedKeys: ['learningRate', 'gamma', 'rewardGoal'],
    current: null,
    best: null,
    history: [],
    message: '自動探索はまだ実行していません',
    ...patch,
  }
}

const RUNNING = message({
  running: true,
  phase: 'running',
  studyName: 'live-ginza',
  trial: 12,
  trialProgress: 0.42,
  finishedTrials: 4,
  priorTrials: 8,
  current: { learningRate: 2e-4 },
  best: { trial: 9, score: 0.3125, params: { learningRate: 1.8e-4 } },
  history: [
    { trial: 10, score: 0.2, outcome: 'complete' },
    { trial: 11, score: -0.42, outcome: 'pruned' },
  ],
})

function render(tune: Msg | null, ctx: Ctx = READY): string {
  return renderToStaticMarkup(createElement(AutoTuneCard, { tune, onToggle: () => {}, ...ctx }))
}

function toggle(html: string): string {
  const m = html.match(/<input[^>]*role="switch"[^>]*>/)
  assert.ok(m, html)
  return m[0]
}

describe('AutoTuneCard', () => {
  test('止まっていて始められるときは、押せる OFF のトグルでバッジは出ない', () => {
    const html = render(message())
    assert.ok(!/checked/.test(toggle(html)))
    assert.ok(!/disabled/.test(toggle(html)))
    assert.ok(!html.includes('自動調整中'), html)
    assert.ok(html.includes('--m3-bar-value:0'), html)
  })

  test('探索中は ON・「自動調整中」・進み具合・最良・直近の試行を出す', () => {
    const html = render(RUNNING)
    assert.ok(/checked/.test(toggle(html)))
    assert.ok(!/disabled/.test(toggle(html)), '止めるほうはいつでも押せる')
    assert.ok(html.includes('自動調整中'), html)
    assert.match(html, /data-testid="autotune-status">試行 #12 を走らせています（42%）</)
    assert.ok(html.includes('--m3-bar-value:0.42'), 'バーは幅ではなく scaleX の変数で伸ばす')
    assert.match(html, /data-testid="autotune-best">試行 #9（スコア \+0\.313）</)
    assert.match(html, /data-testid="autotune-trials">この探索で 4 試行（前回までの完了 8 件）</)
    assert.ok(html.includes('#11 -0.420'), html)
    assert.ok(!/width:\s*42%/.test(html))
  })

  test('Optuna が無いときは押せず、理由を出す', () => {
    const html = render(message({ available: false }))
    assert.ok(/disabled/.test(toggle(html)))
    assert.match(html, /data-testid="autotune-blocked">Optuna が入っていません/)
  })

  test('接続していない・マップが無い・実用モード・認識器の学習中は始められない', () => {
    for (const [ctx, word] of [
      [{ ...READY, connected: false }, '接続していません'],
      [{ ...READY, mapLoaded: false }, 'マップを読み込んで'],
      [{ ...READY, practical: true }, '実用モード'],
      [{ ...READY, suspended: true }, '認識器の学習中'],
    ] as const) {
      const html = render(message(), ctx)
      assert.ok(/disabled/.test(toggle(html)), word)
      assert.ok(html.includes(word), `${word}: ${html}`)
    }
  })

  test('サーバーの状態が届く前（古いサーバー）は押せず、例外なく描ける', () => {
    const html = render(null)
    assert.ok(/disabled/.test(toggle(html)))
    assert.ok(!html.includes('NaN'), html)
  })

  test('直近の試行は 8 件までで、スコアが無い試行は — にする', () => {
    const history = Array.from({ length: 12 }, (_, i) => ({ trial: i, score: i === 11 ? null : 0.1, outcome: 'complete' as const }))
    const view = store.autotuneView(message({ running: true, phase: 'waiting', history }), READY)
    assert.equal(view.history.length, store.AUTOTUNE_HISTORY_SHOWN)
    assert.equal(view.history[0].trial, 4)
    assert.equal(view.history.at(-1)?.text, '#11 —')
    assert.equal(view.statusText, '次の試行のパラメータを選んでいます')
    assert.equal(view.progress, 0)
  })
})

describe('Slider の auto', () => {
  test('自動の間は動かせず、印と値の入れ替え表示を出す', () => {
    const html = renderToStaticMarkup(
      createElement(Slider, { label: '学習率', value: 2e-4, min: 1e-5, max: 1e-3, auto: true, onChange: () => {} }),
    )
    assert.ok(html.includes('data-auto="true"'), html)
    assert.ok(html.includes('m3-slider-auto'), html)
    assert.ok(html.includes('m3-valueflash'), html)
    assert.match(html, /<input[^>]*disabled/)
  })

  test('auto でなければ従来どおり', () => {
    const html = renderToStaticMarkup(
      createElement(Slider, { label: '学習率', value: 2e-4, min: 1e-5, max: 1e-3, onChange: () => {} }),
    )
    assert.ok(!html.includes('data-auto'), html)
    assert.ok(!html.includes('m3-slider-auto'), html)
    assert.doesNotMatch(html, /<input[^>]*disabled/)
  })
})

describe('store/autotune', () => {
  test('探索中の tunedKeys のキーだけが自動', () => {
    assert.equal(store.isAutotuned(RUNNING, 'learningRate'), true)
    assert.equal(store.isAutotuned(RUNNING, 'simSpeed'), false)
    assert.equal(store.isAutotuned(message(), 'learningRate'), false)
    assert.equal(store.isAutotuned(null, 'learningRate'), false)
  })

  test('前回は動いていたのに止まっていれば知らせる', () => {
    assert.match(store.lostSessionNotice(true, message()) ?? '', /best_tuned_policy\.pt/)
    assert.equal(store.lostSessionNotice(true, RUNNING), null)
    assert.equal(store.lostSessionNotice(false, message()), null)
    assert.equal(store.lostSessionNotice(null, message()), null)
  })

  test('トグルの状態は driverl_auto_tune_active に控え、読めなくても落ちない', () => {
    const data = new Map<string, string>()
    const fake = {
      getItem: (k: string) => data.get(k) ?? null,
      setItem: (k: string, v: string) => void data.set(k, v),
    } as unknown as Storage
    assert.equal(store.readAutotuneFlag(fake), null)
    store.writeAutotuneFlag(true, fake)
    assert.equal(data.get('driverl_auto_tune_active'), 'true')
    assert.equal(store.readAutotuneFlag(fake), true)
    store.writeAutotuneFlag(false, fake)
    assert.equal(store.readAutotuneFlag(fake), false)

    const broken = {
      getItem: () => {
        throw new Error('blocked')
      },
      setItem: () => {
        throw new Error('blocked')
      },
    } as unknown as Storage
    assert.equal(store.readAutotuneFlag(broken), null)
    store.writeAutotuneFlag(true, broken)
  })

  test('モックの探索空間はバックエンドと同じキー', () => {
    assert.deepEqual(MOCK_TUNED_KEYS, [
      'learningRate',
      'gamma',
      'clipRange',
      'entropyCoef',
      'rewardGoal',
      'rewardCollision',
      'rewardProgress',
      'rewardOffroad',
      'rewardSignal',
      'rewardOverspeed',
      'rewardTime',
    ])
  })
})
