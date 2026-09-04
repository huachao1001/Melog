/** ECharts 图表管理：建卡、增删数据、主题重建。
 *
 * 指标名支持层级命名（如 recall/class_0）：最后一个 '/' 之前为分组名，
 * 同组指标合并为一张多系列卡片（多分类逐类曲线一图对比），legend 显示
 * 去掉分组前缀后的系列名；无 '/' 的指标保持单系列卡片，外观与旧版一致。
 *
 * 大类别分区（train/val/test）：后端显式声明类别集合，指标名首段命中
 * 类别时归入该分区的垂直分块（分区内仍按上述规则分卡），未命中走
 * 原有分组——类别靠显式声明识别，不靠命名猜测。
 *
 * 配色按名称 hash（FNV-1a）从调色板选取：同一指标名恒定同色（刷新/
 * 重建后不变），不同卡片/系列颜色错开；组内碰撞时向后顺延避免同卡撞色。
 */
import { PointDownsampler } from './downsample.js';

// 图表数值显示：最多小数点后 3 位，尾随 0 省略（0.5 不显示成 0.500）
const fmt3 = (v) => {
  const s = Number(v).toFixed(3).replace(/\.?0+$/, '');
  return s === '-0' ? '0' : s;
};

export class ChartManager {
  constructor({ palette, maxPoints, themeProvider }) {
    this.palette = palette;
    this.themeProvider = themeProvider;  // () => { axis, split } 当前主题坐标轴配色
    this.downsampler = new PointDownsampler(maxPoints);
    this.charts = {};   // 卡片键 -> echarts 实例
    this.data = {};     // 卡片键 -> { 完整指标名 -> [{step, value}] }
    this.colors = {};   // 指标名 -> 用户指定颜色（覆盖 hash 自动配色）
    this.categories = new Set();  // 大类别（train/val/test）：命中首段前缀的指标归入分区
    this.sections = {}; // 卡片键 -> 所属分区名（重建时恢复分区归属）
    window.addEventListener('resize', () => this.resizeAll());
  }

  /** 处理 WebSocket 消息：history 全量替换 / update 增量追加 / colors 用户配色 / categories 大类别。 */
  handle(msg) {
    if (msg.type === 'history') {
      if (Array.isArray(msg.categories)) this.#setCategories(msg.categories);
      for (const [name, pts] of Object.entries(msg.metrics)) this.upsert(name, pts);
    } else if (msg.type === 'update') {
      for (const [name, value] of Object.entries(msg.metrics)) this.append(name, msg.step, value, msg.epoch);
    } else if (msg.type === 'categories') {
      if (Array.isArray(msg.categories)) this.#setCategories(msg.categories);
    } else if (msg.type === 'colors') {
      this.setColors(msg.colors);
    }
  }

  /** 更新大类别集合；集合变化会改变指标名到分区的归属，已建卡片按新归属重建。 */
  #setCategories(list) {
    const before = this.categories.size;
    for (const c of list) this.categories.add(c);
    if (this.categories.size !== before) this.regroupAll();
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

  /** 指标名 -> { 卡片键, 分区名 }。
   *
   * 首段命中大类别（train/val/test）时归入该分区的垂直分块：卡片键 =
   * "类别/指标前缀"（分区隔离同名卡片），分区名用于建分区容器；
   * 未命中保持原行为（按最后一个 '/' 前缀分组，无分区）。
   */
  #layoutOf(name) {
    const i = name.indexOf('/');
    if (i > 0) {
      const section = name.slice(0, i);
      if (this.categories.has(section)) {
        const rest = name.slice(i + 1);
        return { card: `${section}/${this.#groupOf(rest)}`, section };
      }
    }
    return { card: this.#groupOf(name), section: null };
  }

  /** 系列显示名：分区指标去掉类别前缀后，再去掉卡片分组前缀。 */
  #seriesLabel(name) {
    const i = name.indexOf('/');
    if (i > 0 && this.categories.has(name.slice(0, i))) {
      const rest = name.slice(i + 1);
      return this.#labelOf(rest);
    }
    return this.#labelOf(name);
  }

  /** 大类别集合变化后，把已建卡片按新归属重排（清 DOM 重建，保留数据与缩放）。 */
  regroupAll() {
    this.sections = {};
    const zooms = {};
    for (const [group, ch] of Object.entries(this.charts)) {
      const opt = ch.getOption();
      if (opt && opt.dataZoom && opt.dataZoom[0]) zooms[group] = { start: opt.dataZoom[0].start, end: opt.dataZoom[0].end };
      ch.dispose();
      delete this.charts[group];
    }
    document.querySelectorAll('#charts .card, #charts .cat').forEach((el) => el.remove());
    for (const group of Object.keys(this.data)) {
      const section = this.#sectionOfCard(group);
      this.ensureChart(group, section);
      this.#syncSeries(group);
      if (zooms[group]) this.charts[group].dispatchAction({ type: 'dataZoom', start: zooms[group].start, end: zooms[group].end });
    }
  }

  /** 由卡片键反推分区名（键为 "类别/指标前缀" 且首段是已知类别时命中）。 */
  #sectionOfCard(card) {
    const i = card.indexOf('/');
    return i > 0 && this.categories.has(card.slice(0, i)) ? card.slice(0, i) : null;
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

  ensureChart(group, section = null) {
    if (this.charts[group]) return this.charts[group];
    document.getElementById('empty')?.remove();
    let parent = document.getElementById('charts');
    if (section) {
      // 大类别分区：标题 + 独立网格，垂直分块；卡片进分区内的网格
      let block = document.querySelector(`#charts .cat[data-cat="${CSS.escape(section)}"]`);
      if (!block) {
        block = document.createElement('div');
        block.className = 'cat';
        block.dataset.cat = section;
        block.innerHTML = `<h2>${section}</h2><div class="cat-grid"></div>`;
        parent.appendChild(block);
      }
      parent = block.querySelector('.cat-grid');
    }
    this.sections[group] = section;
    const card = document.createElement('div');
    card.className = 'card';
    // 分区卡标题去掉类别前缀（分区标题已含类别）：train/loss 卡显示 "loss"
    const title = section ? group.slice(section.length + 1) : group;
    card.innerHTML = `<h3>${title}</h3><div class="chart"></div>`;
    parent.appendChild(card);
    const el = card.querySelector('.chart');
    const chart = echarts.init(el);
    const t = this.themeProvider();
    chart.setOption(this.#option(group, t));
    this.charts[group] = chart;
    this.data[group] = this.data[group] || {};
    return chart;
  }

  upsert(name, points) {
    const { card, section } = this.#layoutOf(name);
    this.ensureChart(card, section);
    this.data[card][name] = points;
    this.#syncSeries(card);
  }

  append(name, step, value, epoch) {
    const { card, section } = this.#layoutOf(name);
    this.ensureChart(card, section);
    const pts = this.data[card][name] || [];
    pts.push(epoch == null ? { step, value } : { step, value, epoch });
    if (pts.length > this.downsampler.maxPoints) {
      this.data[card][name] = this.downsampler.downsample(pts);
    } else {
      this.data[card][name] = pts;
    }
    this.#syncSeries(card);
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
    // epoch 分界线挂在第一个系列上（多系列共享同一分界），无 epoch 时清除
    const marks = this.#epochMarks(group);
    if (series.length) {
      series[0].markLine = marks.length ? this.#markLineOption(marks) : { data: [] };
    }
    chart.setOption({
      legend: { show: multi, data: names.map((n) => this.#seriesLabel(n)) },
      grid: { top: multi ? 36 : 20 },
      series,
    });
  }

  /** 收集整组数据的 epoch 分界：epoch 值变化处一条竖线（含首个 epoch 起点）。 */
  #epochMarks(group) {
    const all = [];
    for (const arr of Object.values(this.data[group])) {
      for (const p of arr) if (p.epoch != null) all.push(p);
    }
    if (!all.length) return [];
    all.sort((a, b) => a.step - b.step);
    const marks = [];
    for (const p of all) {
      // 相邻同 epoch 的重复分界（多系列/降采样导致 x 略有偏差）只保留一条
      if (!marks.length || marks[marks.length - 1].epoch !== p.epoch) {
        marks.push({ xAxis: p.step, epoch: p.epoch });
      }
    }
    return marks;
  }

  /** epoch 分界竖线：灰色虚线，顶部标注 e<epoch>。 */
  #markLineOption(marks) {
    const t = this.themeProvider();
    return {
      silent: true,
      symbol: 'none',
      animation: false,
      lineStyle: { color: t.axis, type: 'dashed', width: 1, opacity: 0.45 },
      label: { position: 'end', color: t.axis, fontSize: 10, formatter: (p) => `e${p.data.epoch}` },
      data: marks,
    };
  }

  /** 轴触发 tooltip：启用 epoch 时标题显示 "epoch N · step X"。 */
  #tooltip(params) {
    if (!params || !params.length) return '';
    const p0 = params[0];
    const x = Array.isArray(p0.value) ? p0.value[0] : p0.value;
    const ep = params.map((p) => (p.data && p.data.epoch != null ? p.data.epoch : null))
      .find((e) => e != null);
    const title = ep != null ? `epoch ${ep} · step ${fmt3(x)}` : `step ${fmt3(x)}`;
    const lines = params.map((p) => {
      const v = Array.isArray(p.value) ? p.value[1] : p.value;
      return `${p.marker} ${p.seriesName}&nbsp;&nbsp;<b>${fmt3(v)}</b>`;
    });
    return [title, ...lines].join('<br/>');
  }

  #series(group, name, color, multi) {
    const pts = this.data[group][name] || [];
    return {
      id: name,  // 以完整指标名为 id，setOption 按 id 合并，后出现的系列不打乱已有系列
      name: this.#seriesLabel(name),
      type: 'line',
      showSymbol: false,
      smooth: true,
      sampling: 'lttb',
      itemStyle: { color },  // 系列主色：legend 标记圆点与 tooltip 悬浮圆点都用它，保证与线色一致
      lineStyle: { width: multi ? 1.5 : 2, color },
      areaStyle: multi ? { opacity: 0 } : { opacity: 0.1, color },
      // 有 epoch 的点带元信息，tooltip 据此显示 epoch；普通点保持 [x, y]
      data: pts.map((p) => (p.epoch == null ? [p.step, p.value] : { value: [p.step, p.value], epoch: p.epoch })),
    };
  }

  /** 主题切换后销毁重建全部图表，保留缩放状态（分区归属不变）。 */
  rebuildAll() {
    const zooms = {};
    for (const [group, ch] of Object.entries(this.charts)) {
      const opt = ch.getOption();
      if (opt && opt.dataZoom && opt.dataZoom[0]) zooms[group] = { start: opt.dataZoom[0].start, end: opt.dataZoom[0].end };
      ch.dispose();
      delete this.charts[group];
    }
    // 连同旧卡片 / 分区 DOM 一起移除，避免重复建卡
    document.querySelectorAll('#charts .card, #charts .cat').forEach((el) => el.remove());
    for (const group of Object.keys(this.data)) {
      this.ensureChart(group, this.sections[group] ?? this.#sectionOfCard(group));
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
      xAxis: { type: 'value', name: 'step', axisLabel: { color: t.axis, formatter: fmt3 }, axisLine: { lineStyle: { color: t.split } }, splitLine: { lineStyle: { color: t.split } } },
      yAxis: { type: 'value', scale: true, axisLabel: { color: t.axis, formatter: fmt3 }, splitLine: { lineStyle: { color: t.split } } },
      tooltip: { trigger: 'axis', confine: true, formatter: (params) => this.#tooltip(params) },
      // 多系列时 legend 可滚动翻页（类别多也不挤爆卡片）；单系列卡片隐藏。
      // icon 用实心圆点：直接填充系列色（line 系列默认图例是白心圆环）
      legend: {
        show: false,
        type: 'scroll',
        top: 4,
        icon: 'circle',
        itemWidth: 12,
        itemHeight: 12,
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
