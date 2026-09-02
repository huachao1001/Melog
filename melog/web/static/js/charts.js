/** ECharts 图表管理：建卡、增删数据、主题重建。 */
import { PointDownsampler } from './downsample.js';

export class ChartManager {
  constructor({ palette, maxPoints, themeProvider }) {
    this.palette = palette;
    this.themeProvider = themeProvider;  // () => { axis, split } 当前主题坐标轴配色
    this.downsampler = new PointDownsampler(maxPoints);
    this.charts = {};   // metric name -> echarts instance
    this.data = {};     // metric name -> [{step, value}]
    window.addEventListener('resize', () => this.resizeAll());
  }

  /** 处理 WebSocket 消息：history 全量替换 / update 增量追加。 */
  handle(msg) {
    if (msg.type === 'history') {
      for (const [name, pts] of Object.entries(msg.metrics)) this.upsert(name, pts);
    } else if (msg.type === 'update') {
      for (const [name, value] of Object.entries(msg.metrics)) this.append(name, msg.step, value);
    }
  }

  ensureChart(name) {
    if (this.charts[name]) return this.charts[name];
    document.getElementById('empty')?.remove();
    const card = document.createElement('div');
    card.className = 'card';
    const idx = Object.keys(this.charts).length % this.palette.length;
    card.innerHTML = `<h3>${name}</h3><div class="chart"></div>`;
    document.getElementById('charts').appendChild(card);
    const el = card.querySelector('.chart');
    const chart = echarts.init(el);
    const t = this.themeProvider();
    chart.setOption(this.#option(name, idx, t));
    this.charts[name] = chart;
    this.data[name] = this.data[name] || [];
    return chart;
  }

  upsert(name, points) {
    const chart = this.ensureChart(name);
    this.data[name] = points;
    chart.setOption({ series: [{ data: points.map(p => [p.step, p.value]) }] });
  }

  append(name, step, value) {
    const chart = this.ensureChart(name);
    const pts = this.data[name];
    pts.push({ step, value });
    if (pts.length > this.downsampler.maxPoints) {
      this.data[name] = this.downsampler.downsample(pts);
    }
    chart.setOption({ series: [{ data: this.data[name].map(p => [p.step, p.value]) }] });
  }

  /** 主题切换后销毁重建全部图表，保留缩放状态。 */
  rebuildAll() {
    const zooms = {};
    for (const [name, ch] of Object.entries(this.charts)) {
      const opt = ch.getOption();
      if (opt && opt.dataZoom && opt.dataZoom[0]) zooms[name] = { start: opt.dataZoom[0].start, end: opt.dataZoom[0].end };
      ch.dispose();
      delete this.charts[name];
    }
    // 连同旧卡片 DOM 一起移除，避免重复建卡
    document.querySelectorAll('#charts .card').forEach(el => el.remove());
    for (const [name, pts] of Object.entries(this.data)) {
      const chart = this.ensureChart(name);
      chart.setOption({ series: [{ data: pts.map(p => [p.step, p.value]) }] });
      if (zooms[name]) chart.dispatchAction({ type: 'dataZoom', start: zooms[name].start, end: zooms[name].end });
    }
  }

  resizeAll() {
    for (const ch of Object.values(this.charts)) ch.resize();
  }

  #option(name, idx, t) {
    return {
      backgroundColor: 'transparent',
      animation: false,
      grid: { left: 60, right: 20, top: 20, bottom: 60 },
      xAxis: { type: 'value', name: 'step', axisLabel: { color: t.axis }, axisLine: { lineStyle: { color: t.split } }, splitLine: { lineStyle: { color: t.split } } },
      yAxis: { type: 'value', scale: true, axisLabel: { color: t.axis }, splitLine: { lineStyle: { color: t.split } } },
      tooltip: { trigger: 'axis' },
      // 区域选择：滚轮/触控板缩放 + 底部滑块框选，双击复位；预览小图颜色与主图曲线对齐
      dataZoom: [
        { type: 'inside', start: 0, end: 100 },
        {
          type: 'slider', height: 20, bottom: 10, start: 0, end: 100,
          dataBackground: {
            lineStyle: { color: this.palette[idx], opacity: 0.4 },
            areaStyle: { color: this.palette[idx], opacity: 0.15 }
          },
          selectedDataBackground: {
            lineStyle: { color: this.palette[idx], opacity: 0.85 },
            areaStyle: { color: this.palette[idx], opacity: 0.3 }
          },
          fillerColor: this.palette[idx] + '14',
          handleStyle: { color: this.palette[idx] },
          moveHandleStyle: { color: this.palette[idx] },
          emphasis: { moveHandleStyle: { color: this.palette[idx] } }
        }
      ],
      series: [{ name, type: 'line', showSymbol: false, smooth: true, sampling: 'lttb', lineStyle: { width: 2, color: this.palette[idx] }, areaStyle: { opacity: 0.1, color: this.palette[idx] } }]
    };
  }
}
