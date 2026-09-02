/** 文件浏览器弹窗：目录导航、日志加载、实时/历史切换。 */
const ICON_DIR = '<svg class="fb-ico" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#a855f7" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>';
const ICON_LOG = '<svg class="fb-ico" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#ec4899" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/></svg>';

export class FileBrowser {
  constructor() {
    this.modal = document.getElementById('fbModal');
    this.listEl = document.getElementById('fbList');
    this.pathEl = document.getElementById('fbPath');
    this.msgEl = document.getElementById('fbMsg');
    this.cwd = '';      // 当前浏览目录，空表示盘符根层
    this.parent = '';   // 上一级目录，空表示已在根层
  }

  bind() {
    document.getElementById('logBtn').addEventListener('click', () => this.open());
    document.getElementById('fbClose').addEventListener('click', () => this.close());
    this.modal.addEventListener('click', (e) => { if (e.target === this.modal) this.close(); });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') this.close(); });
    document.getElementById('fbUp').addEventListener('click', () => {
      if (this.parent || this.cwd) this.navigate(this.parent);
    });
    this.pathEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') this.navigate(e.target.value.trim());
    });
    document.getElementById('fbHere').addEventListener('click', () => {
      const input = this.pathEl.value.trim();
      if (input) this.load(input);
      else this.msg('请先进入某个目录', 'err');
    });
    document.getElementById('fbLive').addEventListener('click', async () => {
      await fetch('/api/unload', { method: 'POST' });
      this.close();
    });
  }

  open() {
    this.modal.classList.remove('hidden');
    this.navigate(this.cwd);
  }

  close() {
    this.modal.classList.add('hidden');
  }

  msg(text, cls) {
    this.msgEl.textContent = text;
    this.msgEl.className = cls || '';
  }

  static join(cwd, name) {
    return cwd + (/[\\/]$/.test(cwd) ? '' : '/') + name;
  }

  async navigate(path) {
    this.listEl.innerHTML = '<div class="fb-empty">加载中…</div>';
    this.msg('');
    try {
      const res = await fetch('/api/fs?path=' + encodeURIComponent(path || ''));
      const info = await res.json();
      if (!res.ok) throw new Error(info.error || '读取失败');

      if (info.roots) return this.#renderRoots(info.roots);
      if (info.file) return this.load(info.file);
      this.#renderDir(info);
    } catch (err) {
      this.listEl.innerHTML = '';
      this.msg(err.message, 'err');
    }
  }

  #renderRoots(roots) {
    this.cwd = '';
    this.parent = '';
    this.pathEl.value = '';
    this.listEl.innerHTML = '';
    for (const root of roots) {
      this.#addItem(root, 'root', () => this.navigate(root));
    }
  }

  #renderDir(info) {
    this.cwd = info.path;
    this.parent = info.parent || '';
    this.pathEl.value = info.path;
    this.listEl.innerHTML = '';
    if (info.parent) {
      this.#addItem('..', '', () => this.navigate(info.parent));
    }
    for (const name of info.dirs) {
      this.#addItem(name, '', () => this.navigate(FileBrowser.join(info.path, name)));
    }
    for (const name of info.files) {
      const item = document.createElement('div');
      item.className = 'fb-item file';
      item.innerHTML = ICON_LOG + `<span class="fb-name">${name}</span>`;
      item.addEventListener('click', () => this.load(FileBrowser.join(info.path, name)));
      this.listEl.appendChild(item);
    }
    if (!info.files.length) {
      // 目录下没有 .melog 日志时给出明确提示（目录仍可继续浏览）
      const hint = document.createElement('div');
      hint.className = 'fb-empty';
      hint.textContent = '无melog文件';
      this.listEl.appendChild(hint);
    }
  }

  #addItem(name, cls, onClick) {
    const item = document.createElement('div');
    item.className = `fb-item ${cls}`;
    item.innerHTML = (cls === 'file' ? ICON_LOG : ICON_DIR) + `<span class="fb-name">${name}</span>`;
    item.addEventListener('click', onClick);
    this.listEl.appendChild(item);
  }

  /** 加载选中的文件或目录；图表经 WebSocket 广播自动刷新。 */
  async load(path) {
    this.msg('加载中…');
    try {
      const res = await fetch('/api/load', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path })
      });
      const body = await res.json();
      if (!res.ok) throw new Error(body.error || '加载失败');
      this.close();
    } catch (err) {
      this.msg(err.message, 'err');
    }
  }
}
