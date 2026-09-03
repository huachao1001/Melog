/** 媒体视图管理：图像 / 音频卡片，步数滑杆回放。
 *
 * 数据来源：
 * - WebSocket 增量：{type: "image"|"audio", name, step, epoch?, url}
 * - 全量替换：{type: "media_history", media: {image: {name: [entries]}, audio: {…}}}
 *   （建连补发、加载历史日志、回到实时时都会下发）
 *
 * 交互：滑杆在最右端时自动跟随最新条目；往回拨后出现"最新"按钮一键跳回。
 * 图像点击在新页签打开原图。
 */
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

const KINDS = ['image', 'audio'];
const EMPTY_HINT = {
  image: '暂无图像 — 训练中调用 logger.log_image("name", img) 即可展示',
  audio: '暂无音频 — 训练中调用 logger.log_audio("name", wav, sr=16000) 即可展示',
};

/** 秒 -> m:ss。 */
const fmtTime = (s) => (Number.isFinite(s) && s >= 0)
  ? `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`
  : '0:00';

export class MediaManager {
  constructor({ onMedia } = {}) {
    this.onMedia = onMedia;                     // 新数据到达回调（用于页签徽标）
    this.data = { image: {}, audio: {} };       // name -> [{step, epoch?, url, sr?}]
    this.cards = { image: {}, audio: {} };      // name -> DOM 引用
  }

  /** 处理 WebSocket 消息。 */
  handle(msg) {
    if (msg.type === 'image' || msg.type === 'audio') {
      this.#add(msg.type, msg.name, msg);
    } else if (msg.type === 'media_history') {
      this.replaceAll(msg.media || {});
    }
  }

  /** 全量替换（历史日志 / 回到实时）。 */
  replaceAll(media) {
    for (const kind of KINDS) {
      this.data[kind] = {};
      for (const [name, entries] of Object.entries(media[kind] || {})) {
        this.data[kind][name] = entries.slice();
      }
      // 重建该类全部卡片
      for (const name of Object.keys(this.cards[kind])) this.#removeCard(kind, name);
      for (const name of Object.keys(this.data[kind])) this.#updateCard(kind, name);
      this.#syncEmpty(kind);
    }
  }

  #add(kind, name, entry) {
    const arr = this.data[kind][name] || (this.data[kind][name] = []);
    const i = arr.findIndex((e) => e.step === entry.step);
    if (i >= 0) arr[i] = entry;
    else { arr.push(entry); arr.sort((a, b) => a.step - b.step); }
    this.#updateCard(kind, name);
    this.#syncEmpty(kind);
    if (this.onMedia) this.onMedia(kind);
  }

  // ------------------------------------------------------------------ 卡片
  #updateCard(kind, name) {
    const entries = this.data[kind][name];
    if (!entries || !entries.length) return;
    let card = this.cards[kind][name];
    if (!card) {
      card = this.#buildCard(kind, name);
      this.cards[kind][name] = card;
      document.getElementById(kind === 'image' ? 'images' : 'audios').appendChild(card.root);
    }
    const slider = card.slider;
    const wasLatest = Number(slider.value) >= Number(slider.max);  // 先判再改 max
    slider.max = entries.length - 1;
    if (wasLatest) slider.value = slider.max;  // 原在最右端则跟随最新，否则保持用户停留的位置
    card.latest.classList.toggle('off', Number(slider.value) >= Number(slider.max));
    this.#show(kind, name, Number(slider.value));
  }

  #buildCard(kind, name) {
    const root = document.createElement('div');
    root.className = 'card media-card';
    root.innerHTML = `<h3>${esc(name)}</h3>
      <div class="media-stage"></div>
      <div class="media-caption hidden"></div>
      <div class="media-ctrl">
        <input type="range" min="0" max="0" value="0" step="1" aria-label="选择步数">
        <button class="latest off" title="跳到最新">最新</button>
        <span class="media-pos"></span>
      </div>`;
    const card = {
      root,
      stage: root.querySelector('.media-stage'),
      caption: root.querySelector('.media-caption'),
      slider: root.querySelector('input[type="range"]'),
      pos: root.querySelector('.media-pos'),
      latest: root.querySelector('.latest'),
      player: null,
    };
    const { slider } = card;
    if (kind === 'audio') {
      root.classList.add('audio-card');
      this.#buildPlayer(card);
    }
    slider.addEventListener('input', () => {
      const idx = Number(slider.value);
      this.#show(kind, name, idx);
      // 用 off（visibility 占位隐藏）而非 hidden：拖动时按钮显隐不再改变
      // 滑杆宽度，避免"变宽→值变化→再显隐"的抖动循环
      card.latest.classList.toggle('off', idx >= Number(slider.max));
    });
    card.latest.addEventListener('click', () => {
      slider.value = slider.max;
      card.latest.classList.add('off');
      this.#show(kind, name, Number(slider.max));
    });
    if (kind === 'image') {
      card.stage.addEventListener('click', () => {
        const entries = this.data[kind][name];
        const e = entries && entries[Number(slider.value)];
        if (e) window.open(e.url, '_blank');
      });
    }
    return card;
  }

  /** 自绘音频播放器：圆形播放键 + 进度条 + 时间，原生 audio 隐藏在背后。 */
  #buildPlayer(card) {
    const player = document.createElement('audio');
    player.preload = 'metadata';
    card.player = player;

    const wrap = document.createElement('div');
    wrap.className = 'audio-player';
    wrap.innerHTML = `
      <button class="ap-play" title="播放 / 暂停" aria-label="播放或暂停">
        <svg class="icon-play" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
        <svg class="icon-pause" viewBox="0 0 24 24"><path d="M6 5h4v14H6zM14 5h4v14h-4z"/></svg>
      </button>
      <input type="range" class="ap-seek" min="0" max="1000" value="0" step="1" aria-label="播放进度">
      <span class="ap-time">0:00 / 0:00</span>`;
    const seek = wrap.querySelector('.ap-seek');
    const time = wrap.querySelector('.ap-time');

    wrap.querySelector('.ap-play').addEventListener('click', () => {
      if (card.player.paused) card.player.play().catch(() => {});  // 加载失败等场景静默
      else card.player.pause();
    });
    seek.addEventListener('input', () => {
      const d = card.player.duration;
      if (Number.isFinite(d)) card.player.currentTime = (Number(seek.value) / 1000) * d;
    });
    const sync = () => {
      const p = card.player;
      const d = p.duration;
      if (Number.isFinite(d) && d > 0) seek.value = Math.round(((p.currentTime || 0) / d) * 1000);
      time.textContent = `${fmtTime(p.currentTime || 0)} / ${fmtTime(d)}`;
    };
    player.addEventListener('timeupdate', sync);
    player.addEventListener('loadedmetadata', sync);
    player.addEventListener('durationchange', sync);
    player.addEventListener('play', () => wrap.classList.add('playing'));
    player.addEventListener('pause', () => wrap.classList.remove('playing'));
    player.addEventListener('ended', () => wrap.classList.remove('playing'));
    player.addEventListener('error', () => { time.textContent = '加载失败'; });

    card.awrap = wrap;
    card.apseek = seek;
    card.aptime = time;
    wrap.appendChild(player);  // 隐藏的原生元素（css 已 display:none）
    card.stage.appendChild(wrap);
  }

  /** 把第 idx 个条目渲染到卡片（图像换图 / 音频换源 + 位置文案）。 */
  #show(kind, name, idx) {
    const card = this.cards[kind][name];
    const entries = this.data[kind][name];
    if (!card || !entries || !entries[idx]) return;
    const e = entries[idx];
    const pos = e.epoch != null ? `epoch ${e.epoch} · step ${e.step}` : `step ${e.step}`;
    card.pos.textContent = pos;
    // 配文随条目切换：textContent 渲染，用户文本安全；无配文时隐藏
    if (e.caption) {
      card.caption.textContent = e.caption;
      card.caption.classList.remove('hidden');
    } else {
      card.caption.textContent = '';
      card.caption.classList.add('hidden');
    }
    if (kind === 'image') {
      let img = card.stage.querySelector('img');
      if (!img) {
        img = document.createElement('img');
        img.alt = name;
        card.stage.appendChild(img);
      }
      if (img.dataset.url !== e.url) { img.src = e.url; img.dataset.url = e.url; }
    } else if (card.player) {
      if (card.player.dataset.url !== e.url) {
        card.player.pause();
        card.player.src = e.url;
        card.player.dataset.url = e.url;
        card.player.load();
        card.apseek.value = 0;
        card.aptime.textContent = '0:00 / 0:00';
        card.awrap.classList.remove('playing');
      }
    }
  }

  #removeCard(kind, name) {
    const card = this.cards[kind][name];
    if (card) card.root.remove();
    delete this.cards[kind][name];
  }

  /** 容器空态提示：无卡片时显示引导文案。 */
  #syncEmpty(kind) {
    const box = document.getElementById(kind === 'image' ? 'images' : 'audios');
    let empty = box.querySelector('.media-empty');
    const hasCards = Object.keys(this.cards[kind]).length > 0;
    if (!hasCards && !empty) {
      empty = document.createElement('div');
      empty.className = 'media-empty';
      empty.textContent = EMPTY_HINT[kind];
      box.appendChild(empty);
    } else if (hasCards && empty) {
      empty.remove();
    }
  }
}
