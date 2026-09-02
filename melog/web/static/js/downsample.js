/** 指标序列降采样：等宽分桶取均值，保留首尾点（与服务端逻辑一致）。 */
export class PointDownsampler {
  constructor(maxPoints) {
    this.maxPoints = maxPoints;  // 前端缓冲上限
  }

  downsample(points) {
    const maxPoints = this.maxPoints;
    const n = points.length;
    if (!maxPoints || maxPoints < 2 || n <= maxPoints) return points;
    const bucket = n / maxPoints;
    const out = [];
    for (let i = 0; i < maxPoints; i++) {
      const s = Math.floor(i * bucket), e = Math.max(s + 1, Math.floor((i + 1) * bucket));
      let ss = 0, vv = 0;
      for (let j = s; j < e; j++) { ss += points[j].step; vv += points[j].value; }
      out.push({ step: ss / (e - s), value: vv / (e - s) });
    }
    out[0] = points[0];
    out[out.length - 1] = points[points.length - 1];
    return out;
  }
}
