import { createContext, useContext } from 'react'

export type Lang = 'en' | 'ja'

export const LANGUAGES: { id: Lang; label: string }[] = [
  { id: 'en', label: 'English' },
  { id: 'ja', label: '日本語' },
]

/**
 * English source strings. `{name}` marks a value filled in by `t(key, { name })`.
 * Every key must also appear in `ja` below (the type enforces it).
 */
const en = {
  // Shared
  'common.close': 'Close',
  'common.save': 'Save',
  'common.cancel': 'Cancel',
  'common.ok': 'OK',
  'common.remove': 'Remove',
  'common.repair': 'Repair',
  'common.rename': 'Rename',
  'common.edit': 'Edit',
  'common.retry': 'Retry',
  'common.defaults': 'Defaults',
  'common.reset': 'Reset',
  'common.show': 'Show',
  'common.on': 'On',
  'common.off': 'Off',
  'common.left': 'Left',
  'common.right': 'Right',
  'common.up': 'Up',
  'common.down': 'Down',
  'common.yes': 'Yes',
  'common.no': 'No',
  'common.none': 'None',
  'common.undo': 'Undo',
  'common.redo': 'Redo',
  'common.apply': 'Apply',
  'common.name': 'Name',

  // Error prefixes (App run labels)
  'err.model': 'Model',
  'err.character': 'Character',
  'err.reference': 'Reference',
  'err.settings': 'Settings',
  'err.tracking': 'Tracking',
  'err.calibrate': 'Calibrate',
  'err.generate': 'Generate',
  'err.stream': 'Stream',
  'err.virtualCam': 'Virtual camera',
  'err.labFeel': 'Lab feel',
  'err.input': 'Input',
  'err.camera': 'Camera',
  'err.download': 'Download',
  'err.reload': 'Reload',
  'err.language': 'Language',

  // Window caption
  'win.label': 'Window',
  'win.minimize': 'Minimize',
  'win.maximize': 'Maximize',
  'win.close': 'Close',

  // Splash
  'splash.loading': 'Loading resources…',
  'splash.ready': 'Ready',

  // Metric strip
  'metric.fpsTitle': 'Frames shown on the preview per second, including in-betweens',
  'metric.gen': 'Gen',
  'metric.genTitle': 'Frames the model generated per second. FPS should be at least this.',
  'metric.model': 'Model',
  'metric.gpuTitle': 'How busy the graphics card is right now, counting every app (games and OBS too)',

  // Preview stage
  'stage.preview': 'Preview',
  'stage.frozen': 'Frozen',
  'stage.live': 'Live',
  'stage.frozenHint': 'Frozen. Drag points, Shift-drag to pan, scroll to zoom. Double-click to undo.',

  // Rail tabs
  'rail.label': 'Rail',
  'rail.desk': 'Desk',
  'rail.settings': 'Settings',

  // Model
  'model.browse': 'Browse model…',
  'model.label': 'Model',
  'model.none': 'No models',
  'model.local': '{name} (local)',
  'model.notLoaded': 'Not loaded yet — press Start stream to load it.',
  'model.offerNew': 'New on the hub (under 30 days). Download into models/dit.',
  'model.offerAvailable': 'On the hub and not on disk yet.',

  // Reference (developer)
  'ref.imagePath': 'Image path',
  'ref.placeholder': 'Path to reference still',
  'ref.browse': 'Browse…',
  'ref.apply': 'Apply ref',

  // Tracking
  'track.mirror': 'Mirror',
  'track.mirrorTitle': 'Mirror look and head turn',
  'track.start': 'Start tracking',
  'track.stop': 'Stop tracking',
  'track.calibrate': 'Calibrate',
  'track.calibrating': 'Calibrating {pct}%',
  'track.calibrateTitle': 'Capture rest for tracking',
  'track.input': 'Tracking input',
  'track.camera': 'Camera',
  'track.cameraN': 'Camera {n}',
  'track.startLab': 'Start Track Lab to listen for iFacialMocap.',
  'track.copyTitle': 'Copy this PC’s address for iFacialMocap',
  'track.copied': 'Copied',
  'track.thisPc': 'This PC',
  'track.sameWifi': 'Same Wi-Fi as the iPhone.',
  'track.port': 'Port',
  'track.ifmLive': 'Live · {peer}',
  'track.ifmWaiting': 'Waiting for the phone',

  // Feel + Track Lab lamp
  'feel.title': 'Feel',
  'feel.smooth': 'Smooth',
  'feel.smoothTitle': 'Ease each new pose toward the last one. Higher is smoother. Eyes use the same ease.',
  'feel.mouth': 'Mouth',
  'lab.live': 'Track Lab live',
  'lab.connected': 'Track Lab connected',
  'lab.offline': 'Track Lab offline',
  'lab.notRunning': 'Track Lab is not running',

  // Live meters
  'mix.title': 'Live',
  'mix.tracking': 'Tracking',
  'mix.waiting': 'Waiting',
  'mix.eyes': 'Eyes',
  'mix.head': 'Head',
  'mix.mouth': 'Mouth',
  'meter.blinkL': 'blink L',
  'meter.blinkR': 'blink R',
  'meter.lookX': 'look X',
  'meter.lookY': 'look Y',
  'meter.yaw': 'yaw',
  'meter.pitch': 'pitch',
  'meter.roll': 'roll',
  'meter.smile': 'smile',
  'meter.sad': 'sad',
  'meter.A': 'A',
  'meter.I': 'I',
  'meter.U': 'U',
  'meter.E': 'E',

  // Lamp pill
  'lamp.hold': 'hold',
  'lamp.live': 'live',
  'lamp.ok': 'ok',
  'lamp.off': 'off',
  'toggle.status': 'status {light}',

  // Toon (character) block
  'toon.title': 'Toon',

  // Stream
  'stream.title': 'Stream',
  'stream.paused': 'Stream paused',
  'stream.streaming': 'Streaming',
  'stream.vcamOn': 'Virtual camera on',
  'stream.idle': 'Stream idle',
  'stream.loadThenStart': 'Load {model}, then start the stream',
  'stream.stop': 'Stop stream',
  'stream.loadNew': 'Load new model & start',
  'stream.start': 'Start stream',
  'stream.resume': 'Resume',
  'stream.pause': 'Pause',
  'stream.resumeTitle': 'Resume generating frames',
  'stream.pauseTitle': 'Hold the last picture',
  'stream.startCam': 'Start cam',
  'stream.stopCam': 'Stop cam',
  'stream.camTitle': 'Send avatar frames to a virtual camera for OBS',
  'stream.generateOnce': 'Generate once',

  // Settings: language
  'lang.title': 'Language',
  'pick.title': 'Choose your language',
  'pick.hint': 'You can change this later in Settings.',

  // Settings: tune
  'tune.title': 'Tune',
  'tune.defaultsTitle': 'Steps 1, Pose follow 1.0, Reference lock 1.0, Snap 0.58, Hold last on',
  'tune.steps': 'Steps',
  'tune.stepsTitle':
    'Denoise passes per frame. This model is meant for 1 or 2. Each pass sees the pose and the original still together.',
  'tune.poseFollow': 'Pose follow',
  'tune.poseFollowTitle':
    'This 1–2 step model already mixes pose in the joint pass. Fast keeps this at 1.0 — raising it splits pose off the still and melts the face.',
  'tune.refLock': 'Reference lock',
  'tune.refLockTitle': 'This 1–2 step model already mixes the still in the joint pass. Fast keeps this at 1.0.',
  'tune.snap': 'Snap',
  'tune.snapTitle':
    'How much of the new frame is shown while you hold still. Lower blends in the last picture to hide flicker. Default {value}. 1 = no blending. As soon as you move, new frames show in full so the hair keeps up with the face.',
  'tune.holdLast': 'Hold last',
  'tune.holdLastTitle':
    'Start each gen from a light mix of the last picture. Small moves stay consistent; a turn or new character drops the mix so the old face does not stick. Off = every frame is a fresh still from noise.',

  // Settings: performance
  'perf.title': 'Performance',
  'perf.defaultsTitle': 'Max FPS auto, Batch auto, Interpolate on, Inbetweens auto',
  'perf.maxFps': 'Max FPS',
  'perf.maxFpsTitle':
    'Cap on generated frames per second. The GPU idles between frames, so a lower cap leaves headroom for games or OBS. Auto generates just enough frames for the in-betweens to fill the 20 fps preview (1 in-between → 10), so no GPU goes on frames that would never be shown.',
  'perf.auto': 'Auto · {n}',
  'perf.batch': 'Batch',
  'perf.batchTitle':
    'Frames the model draws per call. Bigger gets more frames per second out of the same GPU, but uses more VRAM and adds a little delay, since each call covers a longer stretch of your movement. Auto times this PC the first time and picks the smallest size that keeps up while leaving the GPU room. Each size builds its speed boost once, so changing it takes a moment.',
  'perf.batchRates': 'On this PC: {list}',
  'perf.batchRate': '×{n} {fps} fps',
  'perf.batchRateGuess': '×{n} ≈{fps} fps',
  'perf.batchAuto': 'Auto',
  'perf.batchLocked': 'Stop the stream to change the batch size.',
  'perf.inbetweens': 'Inbetweens',
  'perf.inbetweensTitle':
    'How many extra pictures to print between generated keys. 0 = keys only. Auto fits as many as the 20 fps preview has room for at the speed this PC reaches (more on a slower GPU). Ignored when Interpolate is off.',
  'perf.inbetweensAuto': 'Auto',
  'perf.interpolate': 'Interpolate',
  'perf.interpolateTitle':
    'Print optical-flow frames between DiT keys so motion looks smoother. Off = keys only. Runs off the generate thread so it does not steal Generate FPS.',
  'perf.compile': 'Compile',
  'perf.compileTitle':
    'Speed boost: builds a version of the model tuned for your GPU when the stream starts. The first build can take a minute; off runs at normal speed.',
  'perf.boostBuilding': 'Building the speed boost',

  // Settings: overlay
  'overlay.title': 'Overlay',
  'overlay.outline': 'Outline',
  'overlay.brows': 'Brows',
  'overlay.eyes': 'Eyes',
  'overlay.nose': 'Nose',
  'overlay.mouth': 'Mouth',
  'overlay.iris': 'Iris',
  'overlay.skeleton': 'Skeleton',
  'overlay.hair': 'Hair',

  // Settings: GPU
  'gpu.title': 'GPU',
  'gpu.pickTitle':
    'Which NVIDIA graphics card runs the character. On a PC with two cards you can keep a game on one and VTM Spark on the other.',
  'gpu.auto': 'Automatic',
  'gpu.option': '{n}: {name} · {gb} GB',
  'gpu.inUse': 'Running on {name}',
  'gpu.none': 'No NVIDIA GPU found.',
  'gpu.restartHint': 'The new GPU is used after a restart.',
  'gpu.restart': 'Restart now',
  'gpu.external': 'Set outside the app (CUDA_VISIBLE_DEVICES={value}).',

  // Settings: app
  'app.title': 'App',
  'app.reload': 'Reload backend',
  'app.reloadTitle':
    'Restart Python so code changes load. A small hold window stays up until the desk comes back. Track Lab stays running.',

  // Settings: developer
  'dev.title': 'Developer',
  'dev.trackFps': 'Track FPS',
  'dev.fast': 'Fast',
  'dev.fastOn': 'Fast path on',
  'dev.fastOff': 'Fast off',
  'dev.autoSync': 'Auto sync track',
  'dev.iris': 'Iris',
  'dev.body': 'Body',
  'dev.drivePose': 'Drive pose',

  // Limiters
  'travel.title': 'Limiters',
  'travel.resetTitle': 'Reset head, body, turn, and eye limits',
  'travel.showTitle': 'Draw the head and body walls on the character',
  'travel.head': 'Head',
  'travel.headTitle': 'How far the head may move from rest, in face heights.',
  'travel.headMove': 'Head may move {dir} this far from rest.',
  'travel.body': 'Body',
  'travel.bodyTitle': 'How far the neck, shoulders, and chest may move from rest, in face heights.',
  'travel.bodyMove': 'Body may move {dir} this far from rest.',
  'travel.turnLeft': 'Turn left',
  'travel.turnRight': 'Turn right',
  'travel.tiltLeft': 'Tilt left',
  'travel.tiltRight': 'Tilt right',
  'travel.turnLeftTitle': 'How far the head may turn left on screen.',
  'travel.turnRightTitle': 'How far the head may turn right on screen.',
  'travel.tiltLeftTitle': 'How far the head may tilt left on screen.',
  'travel.tiltRightTitle': 'How far the head may tilt right on screen.',
  'travel.look': 'Look',
  'travel.lookTitle': 'Look angles, eyes, and size.',
  'travel.lookUp': 'Look up',
  'travel.lookUpTitle': 'How far the head may tip up toward the ceiling. Applies while tracking is on.',
  'travel.lookDown': 'Look down',
  'travel.eyes': 'Eyes',
  'travel.eyesTitle': 'How far the pupils may travel inside each eye.',
  'travel.size': 'Size',
  'travel.sizeTitle': 'How much stepping toward or away from the camera may grow or shrink the character.',
  'travel.value': '{label} value',

  // Character library
  'lib.revealFailed': 'Could not open the folder',
  'lib.renameTitle': 'Rename {name}',
  'lib.removeTitle': 'Remove {name}?',
  'lib.removeCurrent': 'Off the desk and out of the library.',
  'lib.removeOther': 'Out of the library.',
  'lib.keep': 'Keep',
  'lib.repairTitle': 'Repair {name}?',
  'lib.repairHint': 'Blend shapes do not match the current plan.',
  'lib.notNow': 'Not now',
  'lib.openCharacters': '{name}. Open characters',
  'lib.emptyAria': 'No character. Click to add',
  'lib.noCharacter': 'No character',
  'lib.clickToAdd': 'Click to add',
  'lib.characters': 'Characters',
  'lib.incompatible': 'Incompatible',
  'lib.actionsFor': 'Actions for {name}',
  'lib.menuAria': '{name} actions',
  'lib.create': 'Create',
  'lib.importTitle': 'Add a character someone shared as a .vtm file',
  'lib.importing': 'Importing…',
  'lib.import': 'Import .vtm',
  'lib.pickVtm': 'Pick a .vtm character file.',
  'lib.imported': 'Imported {name}.',
  'lib.importFailed': 'Import failed: {error}',
  'lib.fitTitle': 'Fit character',
  'lib.creatingTitle': 'Creating character',
  'lib.details': 'Details',
  'lib.showInFolder': 'Show in folder',
  'lib.showInFolderTitle': "Open the folder with this character's .vtm file, ready to share",

  // Character fit
  'fit.hair': 'Hair',
  'fit.points': 'Points',
  'fit.limiters': 'Limiters',
  'fit.tools': 'Fit tools',
  'fit.middle': 'Middle',
  'fit.left': 'Left',
  'fit.right': 'Right',
  'fit.draw': 'Draw',
  'fit.erase': 'Erase',
  'fit.brush': 'Brush',

  // Character details
  'details.title': 'Details · {name}',
  'details.author': 'Author',
  'details.authorPlaceholder': 'Who made this character',
  'details.license': 'License',
  'details.licensePlaceholder': 'e.g. Personal use only',
  'details.description': 'Description',
  'details.otherModel': 'Made with a different model. It will be re-encoded for this model on first load.',
  'details.reading': 'Reading character…',
  'details.unavailable': 'Details are not available for this character.',
  'details.madeWith': 'Made with',
  'details.model': 'Model',
  'details.image': 'Image',
  'details.latent': 'Latent',
  'details.thisDesk': 'This desk',
  'details.sameModel': 'Same model',
  'details.reencode': 'Re-encode on load',
  'details.includes': 'Includes',
  'details.sourceImage': 'Source image',
  'details.poseKeys': 'Pose keys',
  'details.blendShapes': 'Blend shapes',
  'details.hair': 'Hair',
  'details.skeleton': 'Skeleton',
  'details.limiters': 'Limiters',
  'details.file': 'File',
  'details.format': 'Format',
  'details.size': 'Size',
  'details.created': 'Created',
  'details.updated': 'Updated',
  'details.saved': 'Saved.',
  'details.saveFailed': 'Save failed: {error}',
  'details.exportFailed': 'Export failed: {error}',
  'details.busy': 'The desk is busy. Stop streaming to save or export.',
  'details.saving': 'Saving…',
  'details.exporting': 'Exporting…',
  'details.export': 'Export .vtm',
  'details.savedTo': 'Saved {name} to {path}',
  'details.savedName': 'Saved {name}.',
  'details.exportedAs': 'Exported {name} as {file}.',
} as const

export type MessageKey = keyof typeof en

const ja: Record<MessageKey, string> = {
  'common.close': '閉じる',
  'common.save': '保存',
  'common.cancel': 'キャンセル',
  'common.ok': 'OK',
  'common.remove': '削除',
  'common.repair': '修復',
  'common.rename': '名前を変更',
  'common.edit': '編集',
  'common.retry': '再試行',
  'common.defaults': '初期値',
  'common.reset': 'リセット',
  'common.show': '表示',
  'common.on': '有効',
  'common.off': 'オフ',
  'common.left': '左',
  'common.right': '右',
  'common.up': '上',
  'common.down': '下',
  'common.yes': 'はい',
  'common.no': 'いいえ',
  'common.none': 'なし',
  'common.undo': '元に戻す',
  'common.redo': 'やり直す',
  'common.apply': '適用',
  'common.name': '名前',

  'err.model': 'モデル',
  'err.character': 'キャラクター',
  'err.reference': '参照画像',
  'err.settings': '設定',
  'err.tracking': 'トラッキング',
  'err.calibrate': 'キャリブレーション',
  'err.generate': '生成',
  'err.stream': 'ストリーム',
  'err.virtualCam': '仮想カメラ',
  'err.labFeel': '動きの調整',
  'err.input': '入力',
  'err.camera': 'カメラ',
  'err.download': 'ダウンロード',
  'err.reload': '再起動',
  'err.language': '言語',

  'win.label': 'ウィンドウ',
  'win.minimize': '最小化',
  'win.maximize': '最大化',
  'win.close': '閉じる',

  'splash.loading': 'リソースを読み込み中…',
  'splash.ready': '準備完了',

  'metric.fpsTitle': 'プレビューに表示される1秒あたりのフレーム数（中割りを含む）',
  'metric.gen': '生成',
  'metric.genTitle': 'モデルが1秒間に生成したフレーム数。FPS は通常この値以上になります。',
  'metric.model': 'モデル',
  'metric.gpuTitle': 'グラフィックカードの現在の使用率です（ゲームや OBS を含むすべてのアプリの合計）',

  'stage.preview': 'プレビュー',
  'stage.frozen': '固定中',
  'stage.live': 'ライブ',
  'stage.frozenHint':
    '固定中。ポイントはドラッグで移動、Shift+ドラッグで表示位置を移動、スクロールで拡大・縮小。ダブルクリックで元に戻します。',

  'rail.label': 'サイドパネル',
  'rail.desk': 'メイン',
  'rail.settings': '設定',

  'model.browse': 'モデルを選択…',
  'model.label': 'モデル',
  'model.none': 'モデルがありません',
  'model.local': '{name}（ローカル）',
  'model.notLoaded': 'まだ読み込まれていません。「ストリーム開始」を押すと読み込みます。',
  'model.offerNew': 'ハブの新着モデル（30日以内）。models/dit にダウンロードします。',
  'model.offerAvailable': 'ハブで公開中のモデルです（未ダウンロード）。',

  'ref.imagePath': '画像のパス',
  'ref.placeholder': '参照画像のパス',
  'ref.browse': '参照…',
  'ref.apply': '参照画像を適用',

  'track.mirror': '左右反転',
  'track.mirrorTitle': '視線と顔の向きを左右反転します',
  'track.start': 'トラッキング開始',
  'track.stop': 'トラッキング停止',
  'track.calibrate': 'キャリブレーション',
  'track.calibrating': 'キャリブレーション中 {pct}%',
  'track.calibrateTitle': 'トラッキングの基準となる自然な表情を記録します',
  'track.input': 'トラッキング入力',
  'track.camera': 'カメラ',
  'track.cameraN': 'カメラ {n}',
  'track.startLab': 'iFacialMocap を受信するには Track Lab を起動してください。',
  'track.copyTitle': 'iFacialMocap に入力する、このPCのアドレスをコピーします',
  'track.copied': 'コピーしました',
  'track.thisPc': 'このPC',
  'track.sameWifi': 'iPhone と同じ Wi-Fi に接続してください。',
  'track.port': 'ポート',
  'track.ifmLive': '受信中 · {peer}',
  'track.ifmWaiting': 'iPhone からの接続を待っています',

  'feel.title': '動き',
  'feel.smooth': 'なめらかさ',
  'feel.smoothTitle': '新しいポーズを直前のポーズへ徐々に近づけます。値が大きいほどなめらかになります。目にも同じ補正がかかります。',
  'feel.mouth': '口の動き',
  'lab.live': 'Track Lab 稼働中',
  'lab.connected': 'Track Lab 接続済み',
  'lab.offline': 'Track Lab オフライン',
  'lab.notRunning': 'Track Lab が起動していません',

  'mix.title': 'ライブ',
  'mix.tracking': 'トラッキング中',
  'mix.waiting': '待機中',
  'mix.eyes': '目',
  'mix.head': '頭',
  'mix.mouth': '口',
  'meter.blinkL': '瞬き 左',
  'meter.blinkR': '瞬き 右',
  'meter.lookX': '視線 X',
  'meter.lookY': '視線 Y',
  'meter.yaw': 'ヨー',
  'meter.pitch': 'ピッチ',
  'meter.roll': 'ロール',
  'meter.smile': '笑顔',
  'meter.sad': '悲しみ',
  'meter.A': 'あ',
  'meter.I': 'い',
  'meter.U': 'う',
  'meter.E': 'え',

  'lamp.hold': '保留',
  'lamp.live': 'ライブ',
  'lamp.ok': 'OK',
  'lamp.off': 'オフ',
  'toggle.status': '状態: {light}',

  'toon.title': 'キャラクター',

  'stream.title': 'ストリーム',
  'stream.paused': 'ストリーム一時停止中',
  'stream.streaming': 'ストリーム実行中',
  'stream.vcamOn': '仮想カメラ オン',
  'stream.idle': 'ストリーム待機中',
  'stream.loadThenStart': '{model} を読み込んでからストリームを開始します',
  'stream.stop': 'ストリーム停止',
  'stream.loadNew': '新しいモデルで開始',
  'stream.start': 'ストリーム開始',
  'stream.resume': '再開',
  'stream.pause': '一時停止',
  'stream.resumeTitle': 'フレームの生成を再開します',
  'stream.pauseTitle': '最後のフレームで止めます',
  'stream.startCam': 'カメラ開始',
  'stream.stopCam': 'カメラ停止',
  'stream.camTitle': 'アバター映像を仮想カメラに送り、OBS などで使えるようにします',
  'stream.generateOnce': '1枚だけ生成',

  'lang.title': '言語',
  'pick.title': '言語を選択してください',
  'pick.hint': '後から「設定」で変更できます。',

  'tune.title': '調整',
  'tune.defaultsTitle': 'ステップ数 1、ポーズ追従 1.0、参照固定 1.0、反映率 0.58、前フレーム保持 オン',
  'tune.steps': 'ステップ数',
  'tune.stepsTitle':
    '1フレームあたりのノイズ除去の回数です。このモデルは 1 または 2 を想定しています。各パスでポーズと元の画像を同時に参照します。',
  'tune.poseFollow': 'ポーズ追従',
  'tune.poseFollowTitle':
    'この 1〜2 ステップのモデルは、共通のパスの中ですでにポーズを混ぜています。Fast では 1.0 に固定されます。上げるとポーズが元の画像から離れ、顔が崩れます。',
  'tune.refLock': '参照固定',
  'tune.refLockTitle': 'この 1〜2 ステップのモデルは、共通のパスの中ですでに元の画像を混ぜています。Fast では 1.0 に固定されます。',
  'tune.snap': '反映率',
  'tune.snapTitle':
    '静止しているときに新しいフレームをどれだけ表示するかの割合です。下げると前の画像を混ぜてちらつきを抑えます。初期値は {value}。1 = 混ぜません。動いた瞬間から新しいフレームをそのまま表示するので、髪が顔に遅れません。',
  'tune.holdLast': '前フレーム保持',
  'tune.holdLastTitle':
    '直前の画像を少し混ぜた状態から毎回の生成を始めます。小さな動きは安定し、振り向きやキャラクターの切り替え時は混ぜるのをやめるため、前の顔が残りません。オフ = 毎フレームをノイズから新しく生成します。',

  'perf.title': 'パフォーマンス',
  'perf.defaultsTitle': '最大FPS 自動、バッチ 自動、フレーム補間 オン、中割り 自動',
  'perf.maxFps': '最大FPS',
  'perf.maxFpsTitle':
    '1秒あたりに生成するフレーム数の上限です。フレームの合間は GPU が休むため、上限を下げるとゲームや OBS に余裕を残せます。自動では、中割りと合わせて 20 fps のプレビューがちょうど埋まる枚数だけ生成します（中割り 1 → 10）。表示されないフレームに GPU を使いません。',
  'perf.auto': '自動 · {n}',
  'perf.batch': 'バッチ',
  'perf.batchTitle':
    '1回の呼び出しでモデルが描くフレーム数です。増やすと同じ GPU でも1秒あたりのフレームが増えますが、VRAM を多く使い、1回の呼び出しが動きの長い区間を受け持つため少し遅れます。自動では初回にこの PC の速度を計測し、GPU に余裕を残しつつ追いつける最小のサイズを選びます。サイズごとに高速化を一度ビルドするため、変更には少し時間がかかります。',
  'perf.batchRates': 'この PC: {list}',
  'perf.batchRate': '×{n} {fps} fps',
  'perf.batchRateGuess': '×{n} ≈{fps} fps',
  'perf.batchAuto': '自動',
  'perf.batchLocked': 'バッチサイズを変えるにはストリームを停止してください。',
  'perf.inbetweens': '中割り',
  'perf.inbetweensTitle':
    '生成したキーフレームの間に挟む追加フレームの枚数です。0 = キーフレームのみ。自動では、この PC の生成速度で 20 fps のプレビューに収まるだけ挟みます（遅い GPU ほど多く）。フレーム補間がオフのときは無視されます。',
  'perf.inbetweensAuto': '自動',
  'perf.interpolate': 'フレーム補間',
  'perf.interpolateTitle':
    'DiT のキーフレームの間をオプティカルフローで補い、動きをなめらかにします。オフ = キーフレームのみ。生成とは別のスレッドで動くため、生成 FPS は下がりません。',
  'perf.compile': 'コンパイル',
  'perf.compileTitle':
    '高速化：ストリーム開始時に、お使いの GPU 向けに最適化したモデルをビルドします。初回のビルドには1分ほどかかることがあります。オフの場合は通常の速度で動作します。',
  'perf.boostBuilding': '高速化をビルド中',

  'overlay.title': 'オーバーレイ',
  'overlay.outline': '輪郭',
  'overlay.brows': '眉',
  'overlay.eyes': '目',
  'overlay.nose': '鼻',
  'overlay.mouth': '口',
  'overlay.iris': '瞳',
  'overlay.skeleton': '骨格',
  'overlay.hair': '髪',

  'gpu.title': 'GPU',
  'gpu.pickTitle':
    'キャラクターを動かす NVIDIA グラフィックカードを選びます。カードが2枚ある PC では、ゲームと VTM Spark を別々のカードで動かせます。',
  'gpu.auto': '自動',
  'gpu.option': '{n}: {name} · {gb} GB',
  'gpu.inUse': '使用中: {name}',
  'gpu.none': 'NVIDIA GPU が見つかりません。',
  'gpu.restartHint': '新しい GPU は再起動後に使われます。',
  'gpu.restart': '今すぐ再起動',
  'gpu.external': 'アプリの外で設定されています (CUDA_VISIBLE_DEVICES={value})。',

  'app.title': 'アプリ',
  'app.reload': 'バックエンドを再起動',
  'app.reloadTitle':
    'Python を再起動して、コードの変更を読み込みます。画面が戻るまで小さな待機ウィンドウが表示されます。Track Lab は動作したままです。',

  'dev.title': '開発者向け',
  'dev.trackFps': 'トラッキングFPS',
  'dev.fast': '高速モード',
  'dev.fastOn': '高速モード オン',
  'dev.fastOff': '高速モード オフ',
  'dev.autoSync': 'トラッキング自動同期',
  'dev.iris': '瞳',
  'dev.body': '体',
  'dev.drivePose': 'ポーズ駆動',

  'travel.title': '可動域',
  'travel.resetTitle': '頭・体・回転・目の可動域をリセットします',
  'travel.showTitle': 'キャラクターの上に頭と体の可動範囲の枠を表示します',
  'travel.head': '頭',
  'travel.headTitle': '頭が基準の位置から動ける距離です（単位: 顔の高さ）。',
  'travel.headMove': '頭が基準の位置から{dir}へ動ける距離です。',
  'travel.body': '体',
  'travel.bodyTitle': '首・肩・胸が基準の位置から動ける距離です（単位: 顔の高さ）。',
  'travel.bodyMove': '体が基準の位置から{dir}へ動ける距離です。',
  'travel.turnLeft': '左を向く',
  'travel.turnRight': '右を向く',
  'travel.tiltLeft': '左に傾ける',
  'travel.tiltRight': '右に傾ける',
  'travel.turnLeftTitle': '画面上で頭が左を向ける最大の角度です。',
  'travel.turnRightTitle': '画面上で頭が右を向ける最大の角度です。',
  'travel.tiltLeftTitle': '画面上で頭を左に傾けられる最大の角度です。',
  'travel.tiltRightTitle': '画面上で頭を右に傾けられる最大の角度です。',
  'travel.look': '向き',
  'travel.lookTitle': '顔の角度・目・サイズ。',
  'travel.lookUp': '見上げ',
  'travel.lookUpTitle': '頭を上に傾けられる最大の角度です。トラッキング中に適用されます。',
  'travel.lookDown': '見下ろし',
  'travel.eyes': '瞳',
  'travel.eyesTitle': '目の中で瞳が動ける範囲です。',
  'travel.size': 'サイズ',
  'travel.sizeTitle': 'カメラに近づいたり離れたりしたときに、キャラクターを拡大・縮小できる量です。',
  'travel.value': '{label} の値',

  'lib.revealFailed': 'フォルダーを開けませんでした',
  'lib.renameTitle': '「{name}」の名前を変更',
  'lib.removeTitle': '「{name}」を削除しますか？',
  'lib.removeCurrent': '使用中のキャラクターから外し、ライブラリから削除します。',
  'lib.removeOther': 'ライブラリから削除します。',
  'lib.keep': '残す',
  'lib.repairTitle': '「{name}」を修復しますか？',
  'lib.repairHint': 'ブレンドシェイプが現在の構成と一致しません。',
  'lib.notNow': '後で',
  'lib.openCharacters': '{name}。キャラクター一覧を開く',
  'lib.emptyAria': 'キャラクターがありません。クリックして追加',
  'lib.noCharacter': 'キャラクターなし',
  'lib.clickToAdd': 'クリックして追加',
  'lib.characters': 'キャラクター一覧',
  'lib.incompatible': '非対応',
  'lib.actionsFor': '{name} の操作',
  'lib.menuAria': '{name} の操作',
  'lib.create': '新規作成',
  'lib.importTitle': '共有された .vtm ファイルからキャラクターを追加します',
  'lib.importing': '読み込み中…',
  'lib.import': '.vtm を読み込む',
  'lib.pickVtm': '.vtm キャラクターファイルを選んでください。',
  'lib.imported': '「{name}」を読み込みました。',
  'lib.importFailed': '読み込みに失敗しました: {error}',
  'lib.fitTitle': 'キャラクターの調整',
  'lib.creatingTitle': 'キャラクターを作成中',
  'lib.details': '詳細',
  'lib.showInFolder': 'フォルダーに表示',
  'lib.showInFolderTitle': 'このキャラクターの .vtm ファイルがあるフォルダーを開きます（共有用）',

  'fit.hair': '髪',
  'fit.points': 'ポイント',
  'fit.limiters': '可動域',
  'fit.tools': '調整ツール',
  'fit.middle': '中央',
  'fit.left': '左',
  'fit.right': '右',
  'fit.draw': '描く',
  'fit.erase': '消す',
  'fit.brush': 'ブラシ',

  'details.title': '詳細 · {name}',
  'details.author': '作者',
  'details.authorPlaceholder': 'このキャラクターの作者',
  'details.license': 'ライセンス',
  'details.licensePlaceholder': '例: 個人利用のみ',
  'details.description': '説明',
  'details.otherModel': '別のモデルで作成されています。初回の読み込み時に、このモデル用に再エンコードされます。',
  'details.reading': 'キャラクター情報を読み込み中…',
  'details.unavailable': 'このキャラクターの詳細は表示できません。',
  'details.madeWith': '作成環境',
  'details.model': 'モデル',
  'details.image': '画像',
  'details.latent': '潜在表現',
  'details.thisDesk': 'この環境',
  'details.sameModel': '同じモデル',
  'details.reencode': '読み込み時に再エンコード',
  'details.includes': '含まれるデータ',
  'details.sourceImage': '元の画像',
  'details.poseKeys': 'ポーズキー',
  'details.blendShapes': 'ブレンドシェイプ',
  'details.hair': '髪',
  'details.skeleton': '骨格',
  'details.limiters': '可動域',
  'details.file': 'ファイル',
  'details.format': '形式',
  'details.size': 'サイズ',
  'details.created': '作成日時',
  'details.updated': '更新日時',
  'details.saved': '保存しました。',
  'details.saveFailed': '保存に失敗しました: {error}',
  'details.exportFailed': '書き出しに失敗しました: {error}',
  'details.busy': '処理中です。保存や書き出しをするにはストリームを停止してください。',
  'details.saving': '保存中…',
  'details.exporting': '書き出し中…',
  'details.export': '.vtm を書き出す',
  'details.savedTo': '「{name}」を {path} に保存しました',
  'details.savedName': '「{name}」を保存しました。',
  'details.exportedAs': '「{name}」を {file} として書き出しました。',
}

const MESSAGES: Record<Lang, Record<MessageKey, string>> = { en, ja }

/** A string in a given language, whatever the desk is showing (the first-run picker shows both). */
export function messageIn(lang: Lang, key: MessageKey): string {
  return MESSAGES[lang][key]
}

/**
 * Text the backend and Track Lab send in English (progress steps, boot stages,
 * hints, common errors). Anything not listed shows as sent.
 */
const BACKEND_JA: Record<string, string> = {
  // Boot + progress steps
  'Starting…': '起動中…',
  'Loading resources…': 'リソースを読み込み中…',
  Ready: '準備完了',
  Skipped: 'スキップ',
  'Desk API is old — open the UI anyway': 'デスク API が古いバージョンです — このまま画面を開きます',
  'Desk API failed to start': 'デスク API を起動できませんでした',
  'Loading model': 'モデルを読み込み中',
  'Loading image decoder': '画像デコーダーを読み込み中',
  'Loading fast decoder': '高速デコーダーを読み込み中',
  'Compiling model': 'モデルをコンパイル中',
  'Warming up': 'ウォームアップ中',
  'Preparing decoder': 'デコーダーを準備中',
  'Checking the speed boost': '高速化を確認中',
  'Model ready': 'モデルの準備完了',
  'Model failed': 'モデルの読み込みに失敗しました',
  'Moving model to GPU': 'モデルを GPU に転送中',
  'Downloading model': 'モデルをダウンロード中',
  'Loading character': 'キャラクターを読み込み中',
  'Loading character on the new model': '新しいモデルでキャラクターを読み込み中',
  'No character': 'キャラクターなし',
  'Character failed': 'キャラクターの読み込みに失敗しました',
  'Creating character…': 'キャラクターを作成中…',
  'Saving character': 'キャラクターを保存中',
  'Reading character': 'キャラクターファイルを読み込み中',
  'Placing character': 'キャラクターを配置中',
  'Checking blend shapes': 'ブレンドシェイプを確認中',
  'Encoding reference…': '参照画像をエンコード中…',
  'Encoding character': 'キャラクターをエンコード中',
  'Fitting pose': 'ポーズを推定中',
  'Fitting overlay…': 'オーバーレイを調整中…',
  'Loading face detector': '顔検出を読み込み中',
  'Loading iris': '瞳トラッキングを読み込み中',
  'Loading skeleton': '骨格トラッキングを読み込み中',
  'Loading tracker': 'トラッカーを読み込み中',
  'Loading tracker…': 'トラッカーを読み込み中…',
  'Loading Track Lab tracker…': 'Track Lab のトラッカーを読み込み中…',
  'Preparing stream': 'ストリームを準備中',
  'Starting stream': 'ストリームを開始中',
  'Connecting Track Lab': 'Track Lab に接続中',
  'Track Lab connected': 'Track Lab に接続しました',
  'Track Lab still starting': 'Track Lab を起動中',

  // Speed boost (Compile light)
  'Speed boost on': '高速化 オン',
  'Speed boost off': '高速化 オフ',
  'Speed boost builds when the stream starts': '高速化はストリーム開始時にビルドされます',
  'Speed boost needs an NVIDIA GPU': '高速化には NVIDIA GPU が必要です',
  'Speed boost unavailable — running at normal speed': '高速化は利用できません — 通常の速度で動作中',
  'Speed boost unavailable — Triton is not installed (run install.bat)':
    '高速化は利用できません — Triton がインストールされていません（install.bat を実行してください）',

  // Model hub badges
  New: '新着',
  Available: '入手可能',

  // Calibration hints (Track Lab)
  'Keep your mouth closed until Rest finishes': '記録が終わるまで口を閉じたままにしてください',
  'Hold a smile until capture finishes': '記録が終わるまで笑顔を保ってください',
  'Hold a frown until capture finishes': '記録が終わるまで悲しい表情を保ってください',
  "Hold 'ah' until capture finishes": '記録が終わるまで「あ」の口を保ってください',
  "Hold 'ee' until capture finishes": '記録が終わるまで「い」の口を保ってください',
  "Hold 'oo' until capture finishes": '記録が終わるまで「う」の口を保ってください',
  "Hold 'eh' until capture finishes": '記録が終わるまで「え」の口を保ってください',

  // Track Lab errors shown on the desk
  'Track Lab is not running': 'Track Lab が起動していません',
  'Track Lab is still starting': 'Track Lab を起動中です',
  'Track Lab is still starting — wait a moment and try again':
    'Track Lab を起動中です。少し待ってからもう一度お試しください',
  'Track Lab tracker is still loading — wait a moment': 'Track Lab のトラッカーを読み込み中です。少しお待ちください',
  'Track Lab tracker is still loading — wait a moment and try again':
    'Track Lab のトラッカーを読み込み中です。少し待ってからもう一度お試しください',
  'No face while calibrating — hold the pose and try again':
    'キャリブレーション中に顔が検出されませんでした。姿勢を保ったまま、もう一度お試しください',
  'Start tracking first, then hold the face and calibrate':
    '先にトラッキングを開始し、顔を動かさずにキャリブレーションしてください',
  'Track a face first so rest exists': '先に顔をトラッキングして、基準の表情を作成してください',
  'Stop OSF before changing cameras': 'カメラを変更する前にトラッキングを停止してください',
  'Stop listening before changing the port': 'ポートを変更する前に受信を停止してください',
  'Start tracking before recording movement': '動きを記録する前にトラッキングを開始してください',
  'Load a reference image before recording movement': '動きを記録する前に参照画像を読み込んでください',
  'Recording had no tracked frames': '記録中にトラッキングできたフレームがありませんでした',

  // Fit skeleton points
  Neck: '首',
  'R shoulder': '右肩',
  'R elbow': '右ひじ',
  'L shoulder': '左肩',
  'L elbow': '左ひじ',
  Chest: '胸',
}

const BACKEND_JA_PATTERNS: [RegExp, string][] = [
  [/^Character: (.+)$/, 'キャラクター: $1'],
  [/^Loading (.+?)(…)?$/, '$1 を読み込み中$2'],
  [/^No camera at index (\d+)$/, 'カメラ $1 が見つかりません'],
  [/^Could not read (.+)$/, '$1 を読み込めませんでした'],
]

function format(text: string, vars?: Record<string, string | number>): string {
  if (!vars) return text
  return text.replace(/\{(\w+)\}/g, (whole, name: string) =>
    name in vars ? String(vars[name]) : whole,
  )
}

/** Translate an English line that came from the backend. Unknown text passes through. */
function translateBackend(lang: Lang, text: string | null | undefined): string {
  const raw = String(text ?? '')
  if (lang === 'en' || !raw) return raw
  const exact = BACKEND_JA[raw.trim()]
  if (exact) return exact
  for (const [pattern, out] of BACKEND_JA_PATTERNS) {
    if (pattern.test(raw)) return raw.replace(pattern, out)
  }
  return raw
}

// Shown under a bar once it has run a few seconds, so a long compile or
// model load reads as busy backstage, not frozen. Flavour only: the real
// step stays in the label above. Both lists line up one-for-one.
export const BACKSTAGE_LINES: Record<Lang, string[]> = {
  en: [
    'Waking up the avatar…',
    'Teaching the eyes to blink…',
    'Placing character on layer 1…',
    'Asking chat to hold on…',
    'Ironing the green screen…',
    'Tuning the hair physics…',
    'Rehearsing the intro wave…',
    'Warming up the vocal cords…',
    'Loading kawaii.dll…',
    'Polishing the cat ears…',
    'Rigging the smile…',
    'Setting the lighting to cozy…',
    'Convincing the GPU it is showtime…',
    'Counting eyelashes…',
    'Finding outfit…',
    'Feeding the tracker a snack…',
    'Stretching before stream…',
    'Straightening the hoodie strings…',
    'Practicing the “otsukare”…',
    'Adding sparkles to the highlights…',
    'Checking the mic is not muted…',
    'Hiding the spaghetti code…',
    'Loading today’s catchphrase…',
    'Finding the good camera angle…',
    'Taping down the cables…',
    'Practicing the head tilt…',
    'Warming up the smile…',
    'Checking the stream title for typos…',
    'Picking the thumbnail face…',
    'Rehearsing the “welcome back”…',
    'Dusting off the webcam…',
    'Adjusting the chair height…',
    'Hiding the messy desk…',
    'Loading the idle bounce…',
    'Adding shine to the eyes…',
    'Tuning the blush…',
    'Warming up the mouth shapes…',
    'Double-checking the overlay…',
    'Clearing the throat…',
    'Straightening the headset…',
    'Choosing the opening line…',
    'Filling up the energy bar…',
    'Syncing lips to the voice…',
    'Practicing the surprised face…',
    'Waving at the early viewers…',
    'Adjusting the key light…',
    'Loading the signature pose…',
    'Hanging the stream banner…',
    'Practicing the laugh…',
    'Spinning up the GPU fans…',
    'Getting into character…',
    'Rehearsing the goodbye wave…',
    'Setting the scene…',
  ],
  ja: [
    'アバターを起こしています…',
    'まばたきの練習中…',
    'キャラクターをレイヤー1に配置中…',
    'チャットに少し待ってもらっています…',
    'グリーンバックのしわを伸ばしています…',
    '髪の揺れを調整中…',
    'オープニングの手振りをリハーサル中…',
    '発声練習中…',
    'kawaii.dll を読み込み中…',
    '猫耳を磨いています…',
    '笑顔をリギング中…',
    'ライティングを落ち着いた雰囲気に設定中…',
    'GPU に本番だと言い聞かせています…',
    'まつ毛を数えています…',
    '衣装を探しています…',
    'トラッカーにおやつをあげています…',
    '配信前のストレッチ中…',
    'パーカーのひもを整えています…',
    '「おつかれ」の練習中…',
    'ハイライトにきらめきを追加中…',
    'マイクがミュートになっていないか確認中…',
    'スパゲッティコードを隠しています…',
    '今日の決めゼリフを読み込み中…',
    'いいカメラアングルを探しています…',
    'ケーブルをテープで留めています…',
    '首かしげの練習中…',
    '笑顔のウォームアップ中…',
    '配信タイトルの誤字をチェック中…',
    'サムネイル用の表情を選んでいます…',
    '「おかえり」をリハーサル中…',
    'Webカメラのほこりを払っています…',
    '椅子の高さを調整中…',
    '散らかった机を隠しています…',
    '待機モーションを読み込み中…',
    '瞳にハイライトを追加中…',
    'ほっぺの赤みを調整中…',
    '口の形をウォームアップ中…',
    'オーバーレイを再確認中…',
    '咳払いしています…',
    'ヘッドセットの位置を直しています…',
    '最初のひと言を考えています…',
    'エネルギーを充電中…',
    '口の動きを声に合わせています…',
    'びっくり顔の練習中…',
    '早めに来てくれた視聴者に手を振っています…',
    'キーライトを調整中…',
    'お決まりのポーズを読み込み中…',
    '配信バナーを掛けています…',
    '笑い方の練習中…',
    'GPU ファンを回しています…',
    'キャラクターになりきっています…',
    'お別れの手振りをリハーサル中…',
    'シーンを準備中…',
  ],
}

export const STORAGE_KEY = 'vtm-spark.language'

export function isLang(value: unknown): value is Lang {
  return value === 'en' || value === 'ja'
}

/** Saved pick if this WebView kept it, else the OS language. The backend copy wins once it answers. */
export function firstLang(): Lang {
  try {
    const saved = window.localStorage.getItem(STORAGE_KEY)
    if (isLang(saved)) return saved
  } catch {
    /* storage can be off in private mode */
  }
  return navigator.language?.toLowerCase().startsWith('ja') ? 'ja' : 'en'
}

export function paintLang(lang: Lang) {
  document.documentElement.lang = lang
}

export type I18n = {
  lang: Lang
  setLang: (lang: Lang) => Promise<void>
  /** True once a language is saved; false on first run; null until the backend answers. */
  picked: boolean | null
  /** UI string by key, with `{name}` values filled in. */
  t: (key: MessageKey, vars?: Record<string, string | number>) => string
  /** Backend / Track Lab text: translated when known, else shown as sent. */
  tr: (text: string | null | undefined) => string
  /** BCP 47 tag for dates and numbers. */
  locale: string
}

export function makeI18n(lang: Lang, setLang: I18n['setLang'], picked: boolean | null = true): I18n {
  const table = MESSAGES[lang]
  return {
    lang,
    setLang,
    picked,
    t: (key, vars) => format(table[key] ?? en[key], vars),
    tr: (text) => translateBackend(lang, text),
    locale: lang === 'ja' ? 'ja-JP' : 'en-US',
  }
}

export const I18nContext = createContext<I18n>(makeI18n('en', async () => {}))

export function useI18n(): I18n {
  return useContext(I18nContext)
}
