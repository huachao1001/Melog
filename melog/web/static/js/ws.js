/** WebSocket 实时连接：自动重连，状态点反馈。 */
export class LiveSocket {
  constructor(onMessage, statusEl) {
    this.onMessage = onMessage;
    this.statusEl = statusEl;
  }

  connect() {
    const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
    ws.onopen = () => { this.statusEl.className = 'conn ok'; this.statusEl.textContent = '已连接'; this.statusEl.title = '已连接'; };
    ws.onclose = () => {
      this.statusEl.className = 'conn off'; this.statusEl.textContent = '未连接'; this.statusEl.title = '未连接';
      setTimeout(() => this.connect(), 3000);
    };
    ws.onmessage = (e) => this.onMessage(JSON.parse(e.data));
  }
}
