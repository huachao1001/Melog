/** ECharts 图表管理：建卡、增删数据、主题重建。
 *
 * 指标名支持层级命名（如 recall/class_0）：最后一个 '/' 之前为分组名，
 * 同组指标合并为一张多系列卡片（多分类逐类曲线一图对比），legend 显示
 * 去掉分组前缀后的系列名；无 '/' 的指标保持单系列卡片，外观与旧版一致。
 *
 * 配色按名称 hash（FNV-1a）从调色板选取：同一指标名恒定同色（刷新/
 * 重建后不变），不同卡片/系列颜色错开；组内碰撞时向后顺延避免同卡撞色。
 */
import { PointDownsampler } from './downsample.js';

export class ChartManager {
  constructor({ palette, maxPoints, themeProvider }) {
    this.palette = palette;
    this.themeProvider = themeProvider;  // () => { axis, split } 当前主题坐标轴配色
    this.downsampler = new PointDownsampler(maxPoints);
    this.charts = {};   // 分组名 -> echarts 实例
    this.data = {};     // 分组名 -> { 完整指标名 -> [{step, value}] }
    this.colors = {};   // 指标名 -> 用户指定颜色（覆盖 hash 自动配色）
    window.addEventListener('resize', () => this.resizeAll());
  }

  /** 处理 WebSocket 消息：history 全量替换 / update 增量追加 / colors 用户配色。 */
  handle(msg) {
    if (msg.type === 'history') {
      for (const [name, pts] of Object.entries(msg.metrics)) this.upsert(name, pts);
    } else if (msg.type === 'update') {
      for (const [name, value] of Object.entries(msg.metrics)) this.append(name, msg.step, value);
    } else if (msg.type === 'colors') {
      this.setColors(msg.colors);
    }
  }

  /** 应用用户指定颜色（增量合并），已建卡片立即重新着色。 */
  setColors(colors) {
    this.colors = { ...this.colors, ...(colors || {}) };
    for (const group of Object.keys(this.charts)) this.#syncSeries(group);
  }

  /** 指标名 -> 卡片分组名：最后一个 '/' 之前的前缀，无 '/' 则自成一组。 */
  #groupOf(name) {
    const i = name.lastIndexOf('/');
    return i > 0 ? name.slice(0, i) : name;
  }

  /** 系列显示名：去掉分组前缀（卡片标题已含分组）。 */
  #labelOf(name) {
    const g = this.#groupOf(name);
    return g === name ? name : name.slice(g.length + 1);
  }

  /** FNV-1a 字符串 hash（32 位无符号），用于按名称稳定选色。
   *
   * 末尾做 xor 折叠（高低位互换）：FNV 的乘法结构会让同前缀名字
   * （如 class_0..3）的低位聚集，折叠后 mod 依赖全部 32 位，分布均匀。
   */
  #hash(str) {
    let h = 0x811c9dc5;
    for (let i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 0x01000193);
    }
    h ^= h >>> 16;
    return h >>> 0;
  }

  /** 按名称 hash 选色：同名恒定同色；used 中已占用的槽位向后顺延。 */
  #colorFor(name, used) {
    const n = this.palette.length;
    let idx = this.#hash(name) % n;
    if (used) {
      for (let k = 0; k < n && used.has(idx); k++) idx = (idx + 1) % n;
      used.add(idx);
    }
    return this.palette[idx];
  }

  ensureChart(group) {
    if (this.charts[group]) return this.charts[group];
    document.getElementById('empty')?.remove();
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = `<h3>${group}</h3><div class="chart"></div>`;
    document.getElementById('charts').appendChild(card);
    const el = card.querySelector('.chart');
    const chart = echarts.init(el);
    const t = this.themeProvider();
    chart.setOption(this.#option(group, t));
    this.charts[group] = chart;
    this.data[group] = this.data[group] || {};
    return chart;
  }

  upsert(name, points) {
    const group = this.#groupOf(name);
    this.ensureChart(group);
    this.data[group][name] = points;
    this.#syncSeries(group);
  }

  append(name, step, value) {
    const group = this.#groupOf(name);
    this.ensureChart(group);
    const pts = this.data[group][name] || [];
    pts.push({ step, value });
    if (pts.length > this.downsampler.maxPoints) {
      this.data[group][name] = this.downsampler.downsample(pts);
    } else {
      this.data[group][name] = pts;
    }
    this.#syncSeries(group);
  }

  /** 组内全部系列按出现顺序写入图表，legend / 网格留白同步增删。 */
  #syncSeries(group) {
    const chart = this.charts[group];
    const names = Object.keys(this.data[group]);
    const multi = names.length > 1 || names[0] !== group;  // 分组型指标（含 '/'）才显示 legend
    const used = new Set();  // 组内颜色去重：hash 碰撞时顺延
    // 用户指定色若是调色板颜色，先占住其槽位，hash 配色自动避开
    for (const n of names) {
      const c = this.colors[n];
      if (c) {
        const i = this.palette.indexOf(c);
        if (i >= 0) used.add(i);
      }
    }
    const series = names.map((n) => {
      const color = this.colors[n] || this.#colorFor(n, used);
      return this.#series(group, n, color, multi);
    });
    chart.setOption({
      legend: { show: multi, data: names.map((n) => this.#labelOf(n)) },
      grid: { top: multi ? 36 : 20 },
      series,
    });
  }

  #series(group, name, color, multi) {
    const pts = this.data[group][name] || [];
    return {
      id: name,  // 以完整指标名为 id，setOption 按 id 合并，后出现的系列不打乱已有系列
      name: this.#labelOf(name),
      type: 'line',
      showSymbol: false,
      smooth: true,
      sampling: 'lttb',
      itemStyle: { color },  // 系列主色：legend 标记圆点与 tooltip 悬浮圆点都用它，保证与线色一致
      lineStyle: { width: multi ? 1.5 : 2, color },
      areaStyle: multi ? { opacity: 0 } : { opacity: 0.1, color },
      data: pts.map((p) => [p.step, p.value]),
    };
  }

  /** 主题切换后销毁重建全部图表，保留缩放状态。 */
  rebuildAll() {
    const zooms = {};
    for (const [group, ch] of Object.entries(this.charts)) {
      const opt = ch.getOption();
      if (opt && opt.dataZoom && opt.dataZoom[0]) zooms[group] = { start: opt.dataZoom[0].start, end: opt.dataZoom[0].end };
      ch.dispose();
      delete this.charts[group];
    }
    // 连同旧卡片 DOM 一起移除，避免重复建卡
    document.querySelectorAll('#charts .card').forEach((el) => el.remove());
    for (const group of Object.keys(this.data)) {
      this.ensureChart(group);
      this.#syncSeries(group);
      if (zooms[group]) this.charts[group].dispatchAction({ type: 'dataZoom', start: zooms[group].start, end: zooms[group].end });
    }
  }

  resizeAll() {
    for (const ch of Object.values(this.charts)) ch.resize();
  }

  #option(group, t) {
    // 卡片主题色也由组名 hash 决定：不同卡片色彩错开且稳定
    const accent = this.#colorFor(group);
    return {
      backgroundColor: 'transparent',
      animation: false,
      grid: { left: 60, right: 20, top: 36, bottom: 60 },
      xAxis: { type: 'value', name: 'step', axisLabel: { color: t.axis }, axisLine: { lineStyle: { color: t.split } }, splitLine: { lineStyle: { color: t.split } } },
      yAxis: { type: 'value', scale: true, axisLabel: { color: t.axis }, splitLine: { lineStyle: { color: t.split } } },
      tooltip: { trigger: 'axis' },
      // 多系列时 legend 可滚动翻页（类别多也不挤爆卡片）；单系列卡片隐藏
      legend: {
        show: false,
        type: 'scroll',
        top: 4,
        textStyle: { color: t.axis, fontSize: 11 },
        pageIconColor: t.axis,
        pageIconInactiveColor: t.split,
        pageTextStyle: { color: t.axis },
      },
      // 区域选择：滚轮/触控板缩放 + 底部滑块框选，双击复位
      dataZoom: [
        { type: 'inside', start: 0, end: 100 },
        {
          type: 'slider', height: 20, bottom: 10, start: 0, end: 100,
          dataBackground: {
            lineStyle: { color: accent, opacity: 0.4 },
            areaStyle: { color: accent, opacity: 0.15 }
          },
          selectedDataBackground: {
            lineStyle: { color: accent, opacity: 0.85 },
            areaStyle: { color: accent, opacity: 0.3 }
          },
          fillerColor: accent + '14',
          handleStyle: { color: accent },
          moveHandleStyle: { color: accent },
          emphasis: { moveHandleStyle: { color: accent } }
        }
      ],
      series: []
    };
  }
}
