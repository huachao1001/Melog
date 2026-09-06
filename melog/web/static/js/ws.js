/** WebSocket 实时连接：自动重连，状态点反馈。 */
export class LiveSocket {
  /** onState(ok)：连接状态变化回调（true 已连 / false 断开），供页面把
   *  空占位文案换成明确的断连提示（区别于"尚无数据"）。 */
  constructor(onMessage, statusEl, onState = null) {
    this.onMessage = onMessage;
    this.statusEl = statusEl;
    this.onState = onState;
  }

  connect() {
    const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
    ws.onopen = () => {
      this.statusEl.className = 'conn ok'; this.statusEl.textContent = '已连接'; this.statusEl.title = '已连接';
      if (this.onState) this.onState(true);
    };
    ws.onclose = () => {
      this.statusEl.className = 'conn off'; this.statusEl.textContent = '未连接'; this.statusEl.title = '未连接';
      if (this.onState) this.onState(false);
      setTimeout(() => this.connect(), 3000);
    };
    ws.onmessage = (e) => this.onMessage(JSON.parse(e.data));
  }
}
