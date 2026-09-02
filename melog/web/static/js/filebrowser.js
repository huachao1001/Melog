/** 文件浏览器弹窗：目录导航、日志加载、实时/历史切换。 */
const ICON_DIR = '<svg class="fb-ico" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#a855f7" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>';
const ICON_LOG = '<svg class="fb-ico" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#ec4899" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/></svg>';
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

export class FileBrowser {
  constructor() {
    this.modal = document.getElementById('fbModal');
    this.listEl = document.getElementById('fbList');
    this.crumbsEl = document.getElementById('fbCrumbs');
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
    // 点击面包屑空白处 → 切换为可编辑的路径输入框
    this.crumbsEl.addEventListener('click', (e) => {
      if (e.target === this.crumbsEl) this.#editPath();
    });
    document.getElementById('fbHere').addEventListener('click', () => {
      if (this.cwd) this.load(this.cwd);
      else this.msg('请先进入某个目录', 'err');
    });
    document.getElementById('fbLive').addEventListener('click', async () => {
      await fetch('/api/unload', { method: 'POST' });
      this.close();
    });
  }

  open() {
    this.modal.classList.remove('hidden');
    if (this.cwd) {
      this.navigate(this.cwd);
    } else {
      // 首次打开：进入服务进程的当前工作目录
      this.#openDefault();
    }
  }

  async #openDefault() {
    try {
      const res = await fetch('/api/fs');
      const info = await res.json();
      await this.navigate(info.default || '');
    } catch (err) {
      this.navigate('');
    }
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
      // 失败原因直接显示在列表区，避免弹窗一片空白
      const div = document.createElement('div');
      div.className = 'fb-empty';
      div.textContent = err.message === 'Failed to fetch'
        ? '无法连接服务：训练进程是否已停止？'
        : (err.message || '加载失败');
      this.listEl.innerHTML = '';
      this.listEl.appendChild(div);
      this.msg(div.textContent, 'err');
    }
  }

  #renderRoots(roots) {
    this.cwd = '';
    this.parent = '';
    this.#renderCrumbs(null);
    this.listEl.innerHTML = '';
    if (!roots.length) {
      this.listEl.innerHTML = '<div class="fb-empty">未找到可浏览的根目录</div>';
      return;
    }
    for (const root of roots) {
      this.#addItem(root, 'root', () => this.navigate(root));
    }
  }

  #renderDir(info) {
    this.cwd = info.path;
    this.parent = info.parent || '';
    this.#renderCrumbs(info.path);
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
      item.innerHTML = ICON_LOG + `<span class="fb-name">${esc(name)}</span>`;
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

  /** 把面包屑切换成编辑框，回车跳转，Esc/失焦还原。 */
  #editPath() {
    const input = document.createElement('input');
    input.className = 'fb-edit';
    input.type = 'text';
    input.spellcheck = false;
    input.placeholder = '输入路径后回车';
    input.value = this.cwd;
    this.crumbsEl.innerHTML = '';
    this.crumbsEl.appendChild(input);
    input.focus();
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') this.navigate(input.value.trim());
      if (e.key === 'Escape') this.#renderCrumbs(this.cwd || null);
    });
    input.addEventListener('blur', () => this.#renderCrumbs(this.cwd || null));
  }

  /** 渲染可点击的路径面包屑：每段目录一个淡色胶囊，点击跳转到该层级。 */
  #renderCrumbs(path) {
    this.crumbsEl.innerHTML = '';
    if (!path) {
      const chip = document.createElement('span');
      chip.className = 'crumb current';
      chip.textContent = '此电脑';
      this.crumbsEl.appendChild(chip);
      return;
    }
    const parts = path.split(/[\\/]/).filter(Boolean);
    if (!parts.length) {
      // 根目录（POSIX 的 "/"）：单枚不可再上级的胶囊
      const chip = document.createElement('span');
      chip.className = 'crumb current';
      chip.textContent = '/';
      chip.title = path;
      this.crumbsEl.appendChild(chip);
      return;
    }
    let acc = parts[0].endsWith(':') ? parts[0] + '\\' : '/' + parts[0];
    for (let i = 0; i < parts.length; i++) {
      if (i > 0) acc = FileBrowser.join(acc, parts[i]);
      const target = acc;  // 每层独立捕获，避免闭包共享同一个变量
      if (i > 0) {
        const sep = document.createElement('span');
        sep.className = 'crumb-sep';
        sep.textContent = '›';
        this.crumbsEl.appendChild(sep);
      }
      const chip = document.createElement('span');
      chip.className = 'crumb' + (i === parts.length - 1 ? ' current' : '');
      chip.textContent = parts[i];  // 只显示目录名，路径分隔符省略
      chip.title = target;
      chip.addEventListener('click', () => this.navigate(target));
      this.crumbsEl.appendChild(chip);
    }
  }

  #addItem(name, cls, onClick) {
    const item = document.createElement('div');
    item.className = `fb-item ${cls}`;
    item.innerHTML = (cls === 'file' ? ICON_LOG : ICON_DIR) + `<span class="fb-name">${esc(name)}</span>`;
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
