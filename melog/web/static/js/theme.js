/** 主题管理：亮/暗切换 + localStorage 持久化。 */
export class ThemeManager {
  constructor(onChange) {
    this.onChange = onChange;  // 主题变化后的回调（如图表重建）
  }

  get dark() {
    return document.body.classList.contains('dark');
  }

  init() {
    this.set(localStorage.getItem('melog-theme') === 'dark');
  }

  toggle() {
    this.set(!this.dark);
  }

  set(dark) {
    document.body.classList.toggle('dark', dark);
    localStorage.setItem('melog-theme', dark ? 'dark' : 'light');
    this.onChange(dark);
  }
}
