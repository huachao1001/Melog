/** 入口：组装各模块。 */
import { ChartManager } from './charts.js';
import { FileBrowser } from './filebrowser.js';
import { MediaManager } from './media.js';
import { ThemeManager } from './theme.js';
import { LiveSocket } from './ws.js';

const PALETTE = [
  '#a855f7', '#ec4899', '#f59e0b', '#10b981', '#06b6d4', '#ef4444', '#84cc16', '#f97316',
  '#3b82f6', '#14b8a6', '#6366f1', '#d946ef', '#eab308', '#22c55e', '#f43f5e', '#0ea5e9',
];

const charts = new ChartManager({
  palette: PALETTE,
  maxPoints: 2000,
  themeProvider: () => ({
    axis: document.body.classList.contains('dark') ? '#8b8f98' : '#6b7280',
    split: document.body.classList.contains('dark') ? '#23262e' : '#eef0f3',
  }),
});

// 媒体（图像/音频）：当前不在对应页签时来新数据，页签上点亮徽标
const media = new MediaManager({
  onMedia: (kind) => {
    const tab = document.querySelector(`.view-tab[data-view="${kind}s"]`);
    if (tab && !tab.classList.contains('active')) tab.querySelector('.badge')?.classList.remove('hidden');
  },
});

const theme = new ThemeManager(() => charts.rebuildAll());
theme.init();

const fileBrowser = new FileBrowser();
fileBrowser.bind();

new LiveSocket((msg) => { charts.handle(msg); media.handle(msg); }, document.getElementById('status')).connect();

// ---------------------------------------------------------------- 页签切换
const VIEW_IDS = { charts: 'charts', images: 'images', audios: 'audios' };
for (const tab of document.querySelectorAll('.view-tab')) {
  tab.addEventListener('click', () => {
    for (const t of document.querySelectorAll('.view-tab')) {
      t.classList.toggle('active', t === tab);
      if (t !== tab) continue;
      t.querySelector('.badge')?.classList.add('hidden');  // 主动进入后清掉徽标
    }
    const view = tab.dataset.view;
    for (const [key, id] of Object.entries(VIEW_IDS)) {
      document.getElementById(id).classList.toggle('hidden', key !== view);
    }
    if (view === 'charts') charts.resizeAll();
  });
}

document.getElementById('themeBtn').addEventListener('click', () => theme.toggle());
