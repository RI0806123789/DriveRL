/** モックの認識器の学習ジョブ。**実機の段階（採点 → 収集 → 学習 → 完了）をそのまま踏む**（速さは実機より桁違いに速い） */

import type {
  DetectorEvaluation,
  DetectorMessage,
  DetectorRequest,
  StatusPayload,
} from '../../types/protocol.ts'
import { MOCK_PRESETS } from './map.ts'

const MOCK_EVAL_SAMPLES = 400

/** 採点結果の見本。数字は銀座での実測に寄せてある（霧で崩れる形）。 */
function makeMockEvaluation(): DetectorEvaluation {
  const classes = [
    { cls: 0, name: 'TRAFFIC_LIGHT', truth: 231, recall: 0.619, attr: 0.594 },
    { cls: 1, name: 'SPEED_SIGN', truth: 42, recall: 0.524, attr: 0.955 },
    { cls: 2, name: 'VEHICLE', truth: 171, recall: 0.456, attr: 1 },
    { cls: 3, name: 'OBSTACLE', truth: 389, recall: 0.586, attr: 1 },
    { cls: 4, name: 'LANE', truth: 236, recall: 0.665, attr: 1 },
  ]
  const weathers = [
    { name: 'clear', truth: 380, recall: 0.808 },
    { name: 'rain', truth: 371, recall: 0.814 },
    { name: 'fog', truth: 318, recall: 0.16 },
  ]
  return {
    samples: MOCK_EVAL_SAMPLES,
    elapsedSec: 4.3,
    overallRecall: 0.584,
    classes: classes.map((c) => ({
      cls: c.cls,
      name: c.name,
      truth: c.truth,
      matched: Math.round(c.truth * c.recall),
      recall: c.recall,
      attributeTotal: c.attr < 1 ? Math.round(c.truth * c.recall) : 0,
      attributeOk: c.attr < 1 ? Math.round(c.truth * c.recall * c.attr) : 0,
      attributeAccuracy: c.attr,
    })),
    weathers: weathers.map((w) => ({
      name: w.name,
      truth: w.truth,
      matched: Math.round(w.truth * w.recall),
      recall: w.recall,
    })),
    weakest: '（モック）信号機 の成績が最も低い（37%）。霧でも落ちています',
  }
}

/** 何も実行していないときの `detector` メッセージ。 */
function makeIdleDetector(): DetectorMessage {
  return {
    type: 'detector',
    state: 'idle',
    running: false,
    message: '（モック）認識器の学習はまだ実行していません',
    progress: 0,
    collected: 0,
    samples: 0,
    epoch: 0,
    epochs: 0,
    batch: 0,
    batches: 0,
    history: [],
    elapsedSec: 0,
    warning: '',
    paramCount: 0,
    presetName: null,
    request: null,
    evaluation: null,
    dataset: null,
    model: { exists: false, filename: 'detector.keras', sizeBytes: 0, modifiedAt: null, inUse: false },
    datasetFile: { exists: false, sizeBytes: 0, modifiedAt: null },
    limits: {
      samplesMin: 200,
      samplesMax: 4800,
      epochsMin: 1,
      epochsMax: 60,
      batchMin: 8,
      batchMax: 128,
      widthMin: 0.25,
      widthMax: 2.0,
      seedMin: 0,
      seedMax: 999999,
    },
  }
}

/** ジョブが外へ知らせる手段 */
export interface DetectorLink {
  sendDetector(msg: DetectorMessage): void
  sendStatus(patch: Partial<StatusPayload>): void
}

export class MockDetectorJob {
  private state: DetectorMessage = makeIdleDetector()
  private timer: ReturnType<typeof setInterval> | null = null
  private readonly link: DetectorLink

  constructor(link: DetectorLink) {
    this.link = link
  }

  get running(): boolean {
    return this.state.running
  }

  send(): void {
    this.link.sendDetector({ ...this.state })
  }

  /** 1 周 250ms で進める。収集は 1 周で 1/24 ずつ、学習は 1 エポック 8 周 */
  start(request: DetectorRequest): void {
    const preset = MOCK_PRESETS.find((p) => p.id === request.presetId)
    const startedAt = performance.now()
    const collects = request.mode !== 'train'
    const trains = request.mode !== 'collect'

    const evaluates = collects && request.focusWeak
    this.state = {
      ...makeIdleDetector(),
      state: evaluates ? 'evaluating' : collects ? 'collecting' : 'preparing',
      running: true,
      message: evaluates
        ? '（モック）いまの認識器の弱点を測っています'
        : collects
          ? `（モック）${preset?.name ?? 'マップ'} を走らせて教師データを集めています`
          : '（モック）保存済みの教師データを読み込んでいます',
      samples: request.samples,
      epochs: request.epochs,
      batches: 8,
      presetName: preset?.name ?? null,
      request,
    }
    this.link.sendStatus({
      simSuspended: true,
      learning: false,
      suspendReason: '（モック）認識器の学習中はシミュレーションを止めています',
      message: '（モック）認識器の学習を始めました',
    })
    this.send()

    let evaluated = evaluates ? 0 : MOCK_EVAL_SAMPLES
    let collected = collects ? 0 : request.samples
    let epoch = 0
    let batch = 0

    if (this.timer) clearInterval(this.timer)
    this.timer = setInterval(() => {
      const d = this.state
      d.elapsedSec = (performance.now() - startedAt) / 1000

      if (evaluated < MOCK_EVAL_SAMPLES) {
        evaluated = Math.min(MOCK_EVAL_SAMPLES, evaluated + MOCK_EVAL_SAMPLES / 4)
        d.state = 'evaluating'
        d.collected = evaluated
        d.samples = MOCK_EVAL_SAMPLES
        d.progress = evaluated / MOCK_EVAL_SAMPLES
        d.message = `（モック）いまの認識器の弱点を測っています（${evaluated} / ${MOCK_EVAL_SAMPLES} 枚）`
        if (evaluated >= MOCK_EVAL_SAMPLES) {
          d.evaluation = makeMockEvaluation()
          d.samples = request.samples
        }
        this.send()
        return
      }

      if (collected < request.samples) {
        collected = Math.min(request.samples, collected + Math.ceil(request.samples / 24))
        d.state = 'collecting'
        d.collected = collected
        d.progress = collected / request.samples
        d.message = `（モック）教師データを集めています（${collected} / ${request.samples} 枚）`
        if (collected >= request.samples) {
          d.dataset = {
            samples: request.samples,
            objectCellRatio: 0.107,
            objectsPerImage: 5.1,
            classCounts: {
              TRAFFIC_LIGHT: Math.round(request.samples * 1.1),
              SPEED_SIGN: Math.round(request.samples * 0.25),
              VEHICLE: Math.round(request.samples * 0.75),
              OBSTACLE: Math.round(request.samples * 2.0),
              LANE: request.samples,
            },
            weatherCounts: request.weatherMix
              ? {
                  clear: Math.round(request.samples * 0.3),
                  drizzle: Math.round(request.samples * 0.12),
                  rain: Math.round(request.samples * 0.13),
                  fog: Math.round(request.samples * 0.33),
                  heavy_fog: Math.round(request.samples * 0.12),
                }
              : { clear: request.samples },
          }
          d.datasetFile = {
            exists: true,
            sizeBytes: request.samples * 1050,
            modifiedAt: new Date().toLocaleString('sv-SE'),
          }
          if (!trains) return this.finish('done', '（モック）教師データの収集が終わりました')
          d.state = 'training'
          d.progress = 0
          d.message = '（モック）認識器を学習しています'
        }
        return this.send()
      }

      if (!trains) return this.finish('done', '（モック）教師データの収集が終わりました')

      batch += 1
      if (batch > d.batches) {
        batch = 1
        epoch += 1
      }
      if (epoch === 0) epoch = 1
      d.state = 'training'
      d.epoch = epoch
      d.batch = batch
      d.progress = (epoch - 1 + batch / d.batches) / request.epochs
      d.message = `（モック）学習中（${epoch} / ${request.epochs} エポック）`
      if (batch === d.batches) {
        const t = epoch / Math.max(1, request.epochs)
        d.history = [
          ...d.history,
          { epoch, loss: 2.9 * Math.exp(-1.8 * t) + 0.15, valLoss: 3.4 * Math.exp(-1.5 * t) + 0.3 },
        ]
      }
      if (epoch >= request.epochs && batch >= d.batches) {
        d.paramCount = Math.round(152992 * request.width * request.width)
        d.model = {
          exists: true,
          filename: 'detector.keras',
          sizeBytes: Math.round(1965573 * request.width * request.width),
          modifiedAt: new Date().toLocaleString('sv-SE'),
          inUse: true,
        }
        return this.finish('done', `（モック）学習が完了しました（${request.epochs} エポック）`)
      }
      this.send()
    }, 250)
  }

  finish(state: DetectorMessage['state'], message: string): void {
    if (this.timer) clearInterval(this.timer)
    this.timer = null
    this.state = { ...this.state, state, running: false, message, progress: 1 }
    this.send()
    this.link.sendStatus({
      simSuspended: false,
      learning: true,
      suspendReason: '',
      message,
    })
  }

  close(): void {
    if (this.timer) clearInterval(this.timer)
    this.timer = null
  }
}
