/** 入口：组装各模块。 */
import { ChartManager } from './charts.js';
import { FileBrowser } from './filebrowser.js';
import { ThemeManager } from './theme.js';
import { LiveSocket } from './ws.js';

const PALETTE = ['#a855f7', '#ec4899', '#f59e0b', '#10b981', '#06b6d4', '#ef4444', '#84cc16', '#f97316'];

const charts = new ChartManager({
  palette: PALETTE,
  maxPoints: 2000,
  themeProvider: () => ({
    axis: document.body.classList.contains('dark') ? '#8b8f98' : '#6b7280',
    split: document.body.classList.contains('dark') ? '#23262e' : '#eef0f3',
  }),
});

const theme = new ThemeManager(() => charts.rebuildAll());
theme.init();

const fileBrowser = new FileBrowser();
fileBrowser.bind();

new LiveSocket((msg) => charts.handle(msg), document.getElementById('status')).connect();

document.getElementById('themeBtn').addEventListener('click', () => theme.toggle());
