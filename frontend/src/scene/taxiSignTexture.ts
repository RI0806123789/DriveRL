/** タクシーの表示灯と表示板のアトラス（Canvas で描く。フォントはシステムのものだけ）。 */

import * as THREE from 'three'
import {
  SIGN_ROWS,
  SIGN_ROW_DARK,
  SIGN_ROW_LAMP_BODY,
  SIGN_ROW_LAMP_TEXT,
  SIGN_ROW_STATUS,
  TAXI_LAMP_TEXT,
  TAXI_SIGN_STATUSES,
} from './taxiSign'

const CELL_W = 512
const CELL_H = 128

/** 表示灯の地色と文字の色。**実在のタクシー会社の配色に似せない**（白地に濃い灰の文字と、細い帯だけ） */
const LAMP_BODY = '#f2eee3'
const LAMP_INK = '#22262c'
const LAMP_BAND = '#c9a24a'
const DARK = '#15171a'
/** 状態ごとの文字の色（LED の表示器らしく、暗い地に明るい文字） */
const STATUS_INK: Record<string, string> = {
  vacant: '#ff4a3d',
  dispatched: '#ffb02e',
  occupied: '#3fdc84',
  paying: '#4fb2ff',
}

const FONT = '"Hiragino Sans", "Yu Gothic", "Meiryo", "Noto Sans CJK JP", sans-serif'

export function createTaxiSignAtlas(): THREE.CanvasTexture {
  const canvas = document.createElement('canvas')
  canvas.width = CELL_W
  canvas.height = CELL_H * SIGN_ROWS
  const ctx = canvas.getContext('2d')
  if (ctx) {
    const row = (r: number) => r * CELL_H
    ctx.fillStyle = LAMP_BODY
    ctx.fillRect(0, row(SIGN_ROW_LAMP_TEXT), CELL_W, CELL_H)
    ctx.fillStyle = LAMP_BAND
    ctx.fillRect(0, row(SIGN_ROW_LAMP_TEXT) + CELL_H - 16, CELL_W, 9)
    ctx.fillStyle = LAMP_INK
    ctx.font = `700 78px ${FONT}`
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText(TAXI_LAMP_TEXT, CELL_W / 2, row(SIGN_ROW_LAMP_TEXT) + CELL_H / 2 - 6)

    ctx.fillStyle = LAMP_BODY
    ctx.fillRect(0, row(SIGN_ROW_LAMP_BODY), CELL_W, CELL_H)
    ctx.fillStyle = DARK
    ctx.fillRect(0, row(SIGN_ROW_DARK), CELL_W, CELL_H)

    TAXI_SIGN_STATUSES.forEach((s, i) => {
      const y = row(SIGN_ROW_STATUS + i)
      ctx.fillStyle = '#07080a'
      ctx.fillRect(0, y, CELL_W, CELL_H)
      ctx.fillStyle = STATUS_INK[s.status]
      ctx.font = `700 92px ${FONT}`
      ctx.fillText(s.text, CELL_W / 2, y + CELL_H / 2 + 2)
    })
  }
  const texture = new THREE.CanvasTexture(canvas)
  texture.colorSpace = THREE.SRGBColorSpace
  texture.anisotropy = 4
  return texture
}
