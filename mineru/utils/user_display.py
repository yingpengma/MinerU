import time
import sys
import threading
import os


class ProgressDisplayer:
    def __init__(self, enabled=False):
        """
        初始化进度显示器。
        :param enabled: 一个布尔值，由 --user_friendly_progress 参数决定。
        """
        self.enabled = enabled
        self.start_time = time.monotonic() if enabled else None
        self.current_message = ""
        self.is_major_step = False
        self.update_thread = None
        self.stop_update = False
        self.lock = threading.Lock()
        self.terminal_width = self._get_terminal_width()
        self.last_update_time = 0

    def _get_terminal_width(self):
        """
        获取终端宽度，用于清理行显示。
        """
        try:
            return os.get_terminal_size().columns
        except:
            return 80

    def show(self, message: str, is_major_step=False):
        """
        更新当前显示的进度信息。
        """
        if not self.enabled:
            return

        with self.lock:
            self.current_message = message
            self.is_major_step = is_major_step
            self.last_update_time = time.time()
            
            # 如果是第一次调用，启动更新线程
            if self.update_thread is None:
                self.stop_update = False
                self.update_thread = threading.Thread(target=self._update_display, daemon=True)
                self.update_thread.start()
            else:
                # 立即更新一次显示
                self._print_current_status()

    def _print_current_status(self):
        """
        打印当前状态信息。
        """
        if not self.enabled:
            return

        # 计算已用时间
        elapsed_seconds = time.monotonic() - self.start_time
        minutes, seconds = divmod(int(elapsed_seconds), 60)
        
        # 格式化时间戳和消息
        timestamp = f"[已用时 {minutes:02d}:{seconds:02d}]"
        
        # 构建显示消息
        if self.is_major_step:
            display_message = f"🚀 {timestamp} {self.current_message}"
        else:
            display_message = f"   {timestamp} {self.current_message}"
        
        # 清除当前行并打印新消息
        # 使用更可靠的方式清除行
        print(f"\r{' ' * self.terminal_width}\r{display_message}", end="", flush=True)

    def _update_display(self):
        """
        后台线程，每秒更新一次显示。
        """
        while not self.stop_update:
            try:
                # 检查是否需要更新（避免过于频繁的更新）
                current_time = time.time()
                if current_time - self.last_update_time >= 0.5:  # 至少间隔0.5秒
                    self._print_current_status()
                    self.last_update_time = current_time
                time.sleep(0.5)  # 更频繁地检查，但更新间隔更长
            except:
                # 如果出现异常，停止更新
                break

    def success(self, message: str):
        """
        打印最终的成功信息。
        """
        if not self.enabled:
            return

        # 停止更新线程
        with self.lock:
            self.stop_update = True
        
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=1)

        # 计算总用时
        elapsed_seconds = time.monotonic() - self.start_time
        minutes, seconds = divmod(int(elapsed_seconds), 60)
        timestamp = f"[总用时 {minutes:02d}:{seconds:02d}]"

        # 换行并打印成功信息
        print()  # 换行
        print(f"🎉 {timestamp} {message}", file=sys.stdout, flush=True)