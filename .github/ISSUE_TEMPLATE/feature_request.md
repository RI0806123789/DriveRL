---
name: 機能・改善提案 (Feature Request)
about: DriveRL への新機能、アルゴリズム改良、UI 改善を提案します
title: "[FEATURE] "
labels: ["enhancement"]
assignees: ''
---

## 提案の概要

## 解決したい課題・背景

## 提案する仕様・実装イメージ
- **対象**:
  - [ ] 強化学習（PPO・報酬・カリキュラム）
  - [ ] センサー・認識（CNN・天候・カメラ）
  - [ ] シミュレータ環境（歩行者・信号・道路）
  - [ ] 実用モード（配車・AI コンシェルジュ）
  - [ ] フロントエンド UI・3D
- **動作フロー / UI のイメージ**:

## 契約・プロトコルへの影響
- [ ] `docs/protocol.md` の変更が必要
- [ ] `backend/app/contracts.py` の変更が必要
- [ ] `frontend/src/types/protocol.ts` の変更が必要
- [ ] 影響なし（内部実装のみ）

## 性能への影響
- [ ] 1 ステップの予算（50ms）や描画に影響しうる（金沢でも測る必要がある）
- [ ] 影響なし

## 代替案・検討したこと
