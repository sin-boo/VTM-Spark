<p align="center">
  <img src="ui/public/splash-art.png" alt="VTM Spark" width="100%">
</p>

<h1 align="center">VTM Spark</h1>

<p align="center">
  <a href="README.md">English</a> | <b>日本語</b> | <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <b>アニメ画像1枚を、ライブで動く VTuber に。</b><br>
  Webカメラや iPhone でキャラクターを動かし、AI モデルが毎フレームを描画します。OBS / Discord / Zoom からは普通の Webカメラとして認識されます。
</p>

<p align="center">
  <img alt="Windows 10/11" src="https://img.shields.io/badge/Windows-10%20%7C%2011-0078D6">
  <img alt="NVIDIA RTX 30/40/50" src="https://img.shields.io/badge/GPU-NVIDIA%20RTX%2030%20%7C%2040%20%7C%2050-76B900">
  <a href="#ライセンス"><img alt="ライセンス: Apache 2.0（サードパーティ製モデルは各ライセンス）" src="https://img.shields.io/badge/license-Apache%202.0%20%2B%20third--party%20models-blue"></a>
  <a href="https://discord.gg/WEawvVs9KX"><img alt="VTM Spark の Discord に参加" src="https://img.shields.io/badge/Discord-%E5%8F%82%E5%8A%A0%E3%81%99%E3%82%8B-5865F2?logo=discord&logoColor=white"></a>
</p>

---

## デモ

<table>
  <tr>
    <th>静止画を1枚渡すだけで</th>
    <th>VTM Spark が動かします</th>
  </tr>
  <tr>
    <td align="center"><img src="character-blueprint/character-blueprint.png" alt="入力画像：緑背景に胸から上のアニメキャラクター" width="360"></td>
    <td align="center"><img src="docs/media/demo-live.gif" alt="同じキャラクターが顔を向け、うなずき、まばたきし、話している様子（VTM Spark で描画）" width="360"></td>
  </tr>
</table>

<sub>右側のフレームはすべて、左の画像1枚から VTM Spark のモデルが描いたものです。ライブ配信と同じトラッキングパイプライン（顔の向き、うなずき・傾き、まばたき、視線、口の形）で動かしており、入力には実際の顔ではなく、iPhone 形式のスクリプト化したモーションを使っています。まばたきには、Track Lab でこのキャラクター用に作成した「Eye closed」シェイプを使用しています。</sub>

<!-- ライブデモ用の枠：実際のセッションの .mp4 を GitHub の README エディタにドラッグし、表示されたリンクをここに貼り付けてください。 -->

## できること

- **画像1枚から、動くキャラクターを。** リギングも Live2D モデルも不要です。キャラクターの胸から上の画像を用意するだけです。
- **あなたの動きをリアルタイムでトラッキング**します。一般的な Webカメラ（OpenSeeFace）または iFacialMocap を入れた iPhone で、顔の向きと傾き、まばたき、視線、口、眉を追従します。
- **AI モデル（キーポイント駆動の DiT）が毎フレームを描画**します。NVIDIA GPU 上でリアルタイムに動作します。
- **「VTM Spark」という名前の Webカメラとして認識され**、OBS、Discord、Zoom など、カメラを使えるあらゆるアプリで使えます。
- **キャラクターは `.vtm` ファイル1つにまとまる**ので、保存・バックアップ・共有が簡単です。
- **Track Lab** で、口の形、まばたき、頭の可動範囲をキャラクターごとに調整できます。

## しくみ

<p align="center"><img src="docs/media/pipeline.svg" alt="Webカメラまたは iPhone → Track Lab → 37 個のキーポイント → VTM-1.5.1 DiT（キャラクター画像を入力）→ SD VAE → VTM Spark カメラ" width="100%"></p>

あなたの顔は 37 個のキーポイント（顔の輪郭、眉、目、虹彩、鼻、口、上半身）に変換されます。モデル `VTM-1.5.1` が、1枚の参照画像をもとにそのポーズのキャラクターを1ステップで描き直し、VAE がそれを 768 × 768 のフレームに変換します。

## 動作環境

- Windows 10 または 11
- NVIDIA GeForce RTX 30 / 40 / 50 シリーズのグラフィックカードと最新のドライバー（AMD、macOS、Linux は現在未対応）
- 初回のみインターネット接続（ツールとモデルの重みは自動でダウンロードされます。**API キーは不要**です）
- Webカメラ、または iFacialMocap アプリを入れた iPhone

## インストール

1. このリポジトリを**ダウンロード**する（緑の **Code** ボタン → **Download ZIP** を選び、展開します）か、クローンします。
2. **`install.bat` をダブルクリック**します。必要なものをすべてアプリのフォルダー内にセットアップし、最後にインストール結果の概要を表示します。
   - Python（[uv](https://docs.astral.sh/uv/) 経由）を `.venv-build` に。Python 3.10 以降がすでにあればそれを使います
   - UI のビルド専用のポータブル Node.js を `.tools\node` に（既存の Node には影響しません）
   - アプリのウィンドウに使う Microsoft のランタイム WebView2（未導入の場合のみ。Windows 11 には標準で入っています）
   - AI モデル、Track Lab のトラッカー、仮想カメラ
3. **`run.exe` をダブルクリック**（帽子のアイコン）して、デスクを開きます。

> **Windows の確認画面が1回表示されます。** 仮想カメラの追加には一度だけ管理者権限が必要なため、VTM Spark のロゴと「VTM Spark Camera Setup」が表示された確認画面が出ます。**はい**（Yes）をクリックしてください。スキップしてもほかの機能はすべて使えますし、カメラは後からアプリで追加できます。

`install.bat` はいつでも再実行できます。足りないものを修復し、完了済みの部分はスキップします。

最新版にするには、`update.bat` をダブルクリックします。まず GitHub を確認し、新しいバージョンがある場合にだけ VTM Spark を終了します。git で更新を取得し（git がなければ GitHub からダウンロードし）、新しいパッケージ、UI の再ビルド、新しいモデルといった残りのインストールまで自動で行います。インストールが途中で中断された場合は、もう一度 `update.bat` を実行すれば続きから再開します。キャラクターとモデルはそのまま残ります。

VTM Spark を削除するには、同じ場所にある `uninstall.bat` をダブルクリックします。仮想カメラ（管理者の確認が1回あります）、Python 環境、ダウンロードしたモデルとキャッシュを削除します。キャラクターを削除する前には確認があります。最後にフォルダーを削除してください。

## はじめての配信

1. `run.exe` で**デスクを開き**、言語を選びます（あとから「設定」で変更できます）。
2. **キャラクターを追加します。** 空のキャラクタープレビューをクリックし、**新規作成** を押して画像を選びます（下の[キャラクター画像の作り方](#キャラクター画像の作り方)を参照）。準備ができると **キャラクターの調整** が開くので、名前を付けて **保存** します。
3. 「トラッキング入力」で**トラッカーを選びます**。
   - **カメラ**：一覧から Webカメラを選びます。
   - **アイフェイシャルモキャップ**（iFacialMocap）：iPhone を使います（[iPhone トラッキング](#iphone-トラッキング)を参照）。
4. **トラッキング開始** を押し、リラックスした表情で口を閉じたまま正面を見て、**キャリブレーション** を押します。これが基準のポーズになります。
5. **ストリーム開始** を押します。初回は GPU へのモデルの読み込みが行われます。
6. **カメラ開始** を押すと、キャラクターが仮想カメラに送られます。
7. **OBS / Discord / Zoom** で、**VTM Spark** という名前のカメラを選びます。OBS では *映像キャプチャデバイス* ソースを追加し、**VTM Spark** を選んでください。背景は緑なので、*クロマキー* フィルターで透過できます。

少し席を外したいときは、**一時停止** で最後の画像を表示したままにできます。

## キャラクター画像の作り方

モデルは、[`character-blueprint/character-blueprint.png`](character-blueprint/character-blueprint.png) とまったく同じ構図の画像で最もよく動作します。

| このようにする | 避けること |
|---|---|
| 正方形、768 × 768 | 縦長・横長の画像 |
| 単色の緑背景（`#00FF00`） | 部屋の背景、グラデーション、背景に落ちる影 |
| 胸から上、キャラクターは中央、カメラ正面向き、あごは水平 | 頭の傾きや横向き |
| 両目を開け、口を閉じた控えめな笑顔 | 開いた口、ウインク、手や小物の写り込み |
| キャラクターは1人、文字やロゴなし | 透かし、UI、ほかの人物 |

**いちばん簡単な方法：** 画像生成 AI に、ブループリントと自分のキャラクターの2枚の画像を、[`character-blueprint/PROMPT.txt`](character-blueprint/PROMPT.txt) のプロンプトと一緒に渡します。ブループリントが決めるのは構図とポーズだけで、キャラクターの見た目はあなたの画像から反映されます。

## iPhone トラッキング

1. Face ID 対応の iPhone に **iFacialMocap** をインストールします。
2. iPhone と PC を**同じ Wi-Fi** に接続します。
3. デスクで **アイフェイシャルモキャップ**（iFacialMocap）を選びます。**このPC** の IP アドレス（クリックでコピー）とポート（デフォルトは **49983**）が表示されます。
4. iFacialMocap で、その IP アドレスを送信先に設定して送信を開始します。
5. いつもどおり **トラッキング開始** を押し、続けて **キャリブレーション** を押します。

データが届かない場合は、Windows ファイアウォールで **プライベート** ネットワークでの Python の通信を許可し、iPhone の送信先がデスクに表示されたアドレスと一致しているか確認してください。

## キャラクター（`.vtm` ファイル）

キャラクターは、`characters` フォルダー内の1つの `.vtm` ファイルです。画像に加えて、それに合わせて調整したすべてのデータ（髪のマスク、体のポイント、動きの制限、口と目のシェイプ）が入っています。

- **ほかの人のキャラクターを追加：** **.vtm を読み込む** を使うか、`.vtm` ファイルをキャラクタープレビューまたはライブラリにドラッグします。
- **自分のキャラクターを共有：** 右クリックして **フォルダーに表示** を選び、その `.vtm` ファイルを送ります。
- **編集・名前の変更・削除：** キャラクターを右クリックします。
- **「…を修復しますか？ ブレンドシェイプが現在の構成と一致しません。」：** このキャラクターを作成した後に、Track Lab のシェイプが変更されています。**修復** を押すと、現在のシェイプがキャラクターにコピーされます。ほかの人から読み込んだキャラクターは独自のシェイプを保持するため、この確認は表示されません。

## Track Lab（微調整）

デスクは Track Lab のトラッカーをバックグラウンドで動かしているので、配信するだけなら Track Lab を開く必要はありません。細かく調整したいときは、`track_lab\start.bat` をダブルクリックして <http://127.0.0.1:5174> を開きます。

- **Blend**（ブレンド）パネル：顔の動きで駆動されるシェイプです。Rest（基準）、Smile（笑顔）、Sad（悲しみ）、母音 A / I / U / E、まばたき用の **Eye open / Eye closed**（目を開く／目を閉じる）があります。**›** を開いてシェイプを編集し、ポイントをドラッグして **Apply**（適用）を押します。シェイプ間の Blend バーで中間点を追加すると、動きがよりなめらかになります。
- **Limiters**（リミッター）：頭や体が動く・振り向く・うなずく範囲の上限と、目やサイズの制限です。**Fit**（自動調整）で画像から設定します。
- **Tracking**（トラッキング）：口を閉じて正面を見た状態で **Set Rest**（基準を設定）を押します。Response、Smooth、Mouth、Gaze の各スライダーで、キャラクターの追従のしかたを調整します。
- **Gen**（生成）は、現在のポイントでキャラクターを1フレーム描画します。シェイプの確認に便利です。

シェイプの編集は、トラッキングを停止した状態で行います。

## 知っておきたい設定

| 設定 | 内容 |
|---|---|
| **なめらかさ** | 頭の動きをやわらげます。0 で生の動き、値を上げるほどふんわりします。まばたきは即座に反映されます。 |
| **口の動き** | 自分の口の動きにどれだけ強く追従するかを決めます。 |
| **左右反転** | 視線や顔の向きの左右を反転します。 |
| **最大FPS** | フレームレートに上限を設け、ゲームや OBS のために GPU の余裕を残します。 |
| **バッチ** | 1ステップで描画するフレーム数です。FPS は上がりますが、VRAM の使用量が増え、わずかに遅延します。 |
| **中割り** | 描画したフレームの間に追加の画像を挟み、動きをなめらかにします。 |
| **コンパイル** | GPU 向けの高速化です。初回のビルドには1分ほどかかることがあります。 |
| **GPU** | NVIDIA カードが複数ある場合に、使用するカードを選びます（再起動後に反映）。 |

## トラブルシューティング

| 症状 | 対処法 |
|---|---|
| `install.bat` が途中で止まる | インターネット接続を確認してください。プロキシやファイアウォールが nodejs.org、github.com、pypi.org をブロックしている場合や、ウイルス対策ソフトが `.venv-build` をロックしている場合に止まることがあります。そのうえで、もう一度実行してください。 |
| 「NVIDIA GPU が見つかりません」 | VTM Spark には NVIDIA RTX カードと最新のドライバーが必要です。 |
| OBS/Discord で **VTM Spark** カメラが見つからない、または真っ黒 | カメラのセットアップ確認画面で **はい**（Yes）をクリックしてください。アプリのフォルダーを移動した場合は、カメラが新しい場所を参照するよう、`install.bat` をもう一度実行してください。 |
| iPhone：「No packets yet」（まだパケットが届いていません） | 両方を同じ Wi-Fi に接続し、Windows ファイアウォールでプライベートネットワークでの Python の通信を許可したうえで、デスクに表示された IP とポートに送信してください。 |
| 「Port 8780 is already in use」（ポート 8780 は使用中です） | 別のプログラムが Track Lab のポートを使用しています。そのプログラムを閉じてから、もう一度起動してください。 |
| 「キャリブレーション中に顔が検出されませんでした」 | 明るい場所でカメラに顔を向け、もう一度 **キャリブレーション** を押してください。 |
| 起動中と表示されるのにウィンドウが出ない | 残っているプロセスの終了を求められたら許可し、もう一度 `run.exe` を開いてください。 |

## モデルとライセンス

モデルは `install.bat` が自動でダウンロードします。手動で取得する必要はありません。

<p align="center"><img src="docs/media/model-downloads.svg" alt="モデルのダウンロードサイズ：髪のセグメンテーション 432 MB、生成モデル 360 MB、SD VAE 335 MB、アニメ顔のランドマーク 39 MB、体のキーポイント 23 MB、OpenSeeFace 21 MB、tiny VAE 9.8 MB、虹彩 6.4 MB、アニメ顔の検出ボックス 6.2 MB、体のトラッキング 5.8 MB" width="100%"></p>

| モデル | 役割 | サイズ | 保存先 | ライセンス | 商用利用 |
|---|---|---|---|---|---|
| `VTM-1.5.1.pt` | キャラクターを描画（DiT） | 360 MB | `models/dit/` | Apache-2.0（当プロジェクト） | 可 |
| `vtm-fast-decoder.pt` | ライブ配信用の高速な画像デコード | 3.9 MB | `models/decoder/` | 当プロジェクト。`cqyan/hybrid-sd-tinyvae` から蒸留（同モデルのライセンスは明記なし、[サードパーティ表記](THIRD_PARTY_NOTICES.md)） | 不明 |
| `animeseg_hair3.pt` | 髪のパーツを検出し、髪を頭に追従させる | 432 MB | `models/trackers/` | Meta Mask2Former を当プロジェクトでファインチューニング、**CC BY-NC 4.0** | **不可** |
| `dwpose_v2.pt` | 画像上の体のキーポイント | 23 MB | `models/trackers/` | Ultralytics YOLO-pose を当プロジェクトでファインチューニング、**AGPL-3.0** | AGPL の条件に従う |
| `iris_pose.pt` | 画像上の虹彩・瞳孔 | 6.4 MB | `models/trackers/` | Ultralytics YOLO-pose を当プロジェクトでファインチューニング、**AGPL-3.0** | AGPL の条件に従う |
| `pose_landmarker_lite.task` | ライブでの体のトラッキング（MediaPipe） | 5.8 MB | `models/trackers/` | Apache-2.0（Google） | 可 |
| OpenSeeFace（5 ファイル） | Webカメラでの顔トラッキング | 21 MB | `vendor/tools/openseeface/models/` | BSD 2-Clause | 可 |
| `face_yolov8n.pt` | アニメ顔の検出ボックス | 6.2 MB | `models/trackers/` | Apache-2.0（[Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer)） | 可 |
| `mmpose_anime-face_hrnetv2.pth` | アニメ顔のランドマーク | 39 MB | `models/trackers/` | MIT（[hysts/anime-face-detector](https://github.com/hysts/anime-face-detector/releases/tag/v0.0.1)） | 可 |
| `stabilityai/sd-vae-ft-mse` | モデルの出力を画像に変換 | 335 MB | Hugging Face のキャッシュ | MIT | 可 |
| `cqyan/hybrid-sd-tinyvae` | より高速な画像デコード | 9.8 MB | Hugging Face のキャッシュ | 公開元による明記なし | 不明 |

最初の7つは当プロジェクトの Hugging Face リポジトリ [sinBoo1/VTM-Spark](https://huggingface.co/sinBoo1/VTM-Spark) から、残りはそれぞれの公開元から直接ダウンロードされます。合計で約 1.24 GB です。Python と PyTorch は別途ダウンロードされます。

「ストリーム開始」を押したときに DiT の重みがない場合は、起動前にダウンロードされます。

## 開発者向け

- `backend/`：アプリ本体（`python -m backend`）。`backend/packaging/` にはインストール用とランチャー用のスクリプトがあります
- `ui/`：Vite/React 製のデスク
- `track_lab/`：トラッカーと Track Lab の UI（`track_lab/` から `python -m backend.pair`）
- `vendor/`：推論、LivePoser、OpenSeeFace、仮想カメラ
- `models/`：DiT、トラッカー、ダウンロードマップ、リファレンス
- `characters/`：あなたの `.vtm` パック（アプリのルート直下）

テスト（任意）：`python -m pytest backend/tests`、および `track_lab/` から `python -m pytest backend harness`。

ベンダーコードは `vendor/` 以下にコミットされています。`backend\packaging\build.ps1` に `-SyncVendor` を渡すのは、オプションの親モノレポ内で開発していて、ベンダーのコピーを更新する必要がある場合だけにしてください。

## ライセンス

Apache License 2.0 は、VTM Spark のソースコードと、**当プロジェクト**独自の学習・ファインチューニングの成果に適用されます（[LICENSE](LICENSE) を参照）。この許諾はサードパーティのモデルの重みには**適用されず**、後から追加されたモデルにも適用されません。

同梱しているモデルのうち2つは、より厳しい条件の重みをベースにしており、その条件が引き続き適用されます。

- **`animeseg_hair3.pt`**（髪のトラッキング）は Meta の Mask2Former の重みをベースにしています：**CC BY-NC 4.0、非商用利用のみ**。
- **`iris_pose.pt` と `dwpose_v2.pt`** は Ultralytics YOLO-pose の重みをベースにしています：**AGPL-3.0**。

ファイルごとの一覧は、上の[モデルとライセンス](#モデルとライセンス)の表をご覧ください。

サードパーティの重みには、それを公開した公開元の公式ライセンスが適用されます。一覧と入手元の URL：[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。OpenSeeFace のバイナリライブラリに関する表記：`vendor/tools/openseeface/Licenses/`。

この README は英語版の翻訳です。内容に相違がある場合は、英語版の [README.md](README.md) および [LICENSE](LICENSE) が優先されます。
