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
  // 分区集合变化时重建左侧切换栏（train / val / test 等 tab）
  onTabsChange: (tabs) => buildTabbar(tabs),
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

// 连接状态变化：尚无图表（#empty 还在）时把占位文案换成明确的断连提示，
// 避免把"服务已退出 / 网络不通"误读成"还没有数据"
new LiveSocket((msg) => { charts.handle(msg); media.handle(msg); }, document.getElementById('status'),
  (ok) => {
    const empty = document.getElementById('empty');
    if (empty) {
      empty.textContent = ok ? '等待训练指标…'
        : '未连接服务器（服务退出或网络不通）— 请确认服务存活后刷新页面';
    }
  }).connect();

// ---------------------------------------------------------------- 页签切换
const VIEW_IDS = { charts: 'charts', images: 'images', audios: 'audios' };

/** 激活顶部页签对应的视图（曲线 / 图像 / 音频）。 */
function showView(view) {
  for (const tab of document.querySelectorAll('.view-tab')) {
    tab.classList.toggle('active', tab.dataset.view === view);
  }
  for (const [key, id] of Object.entries(VIEW_IDS)) {
    document.getElementById(id).classList.toggle('hidden', key !== view);
  }
  if (view === 'charts') charts.resizeAll();
}

for (const tab of document.querySelectorAll('.view-tab')) {
  tab.addEventListener('click', () => {
    showView(tab.dataset.view);
    tab.querySelector('.badge')?.classList.add('hidden');  // 主动进入后清掉徽标
  });
}

// ---------------------------------------------------------------- 左侧分区切换
const TAB_KEY = 'melog-tab';  // localStorage：记住上次选中的分区

/** 按分区集合重建左侧切换栏：只放 StepsBar 里声明过的动态分区
 *  （按声明顺序），不做任何预定义；分区数 < 2 时隐藏（无切换意义）。
 *  默认选中第一个分区，相同分区的图表在同一界面展示；上次选中仍有效
 *  时恢复（localStorage）。 */
function buildTabbar(tabs) {
  const bar = document.getElementById('tabs');
  bar.classList.toggle('hidden', tabs.length < 2);
  let active = tabs[0];  // 默认第一个分区（声明顺序，通常是 train）
  if (localStorage.getItem(TAB_KEY) && tabs.includes(localStorage.getItem(TAB_KEY))) {
    active = localStorage.getItem(TAB_KEY);  // 恢复上次选中（仍是有效分区时）
  }
  bar.innerHTML = '';
  for (const tab of tabs) {
    const btn = document.createElement('button');
    btn.className = 'side-tab' + (tab === active ? ' active' : '');
    btn.textContent = tab;
    btn.title = `只显示 ${tab} 分区`;
    btn.addEventListener('click', () => {
      charts.setActiveTab(tab);
      localStorage.setItem(TAB_KEY, tab);
      for (const b of bar.querySelectorAll('.side-tab')) b.classList.toggle('active', b === btn);
      showView('charts');  // 切分区即切到曲线视图（分区只作用于曲线）
    });
    bar.appendChild(btn);
  }
  charts.setActiveTab(active);
}

document.getElementById('themeBtn').addEventListener('click', () => theme.toggle());
