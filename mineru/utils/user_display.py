import time
import sys


class ProgressDisplayer:
    def __init__(self, enabled=False):
        """
        初始化进度显示器。
        :param enabled: 一个布尔值，由 --user_friendly_progress 参数决定。
        """
        self.enabled = enabled
        self.start_time = time.monotonic() if enabled else None

    def show(self, message: str, is_major_step=False):
        """
        打印一条用户友好的进度信息。
        """
        if not self.enabled:
            return

        # 计算已用时间
        elapsed_seconds = time.monotonic() - self.start_time
        minutes, seconds = divmod(int(elapsed_seconds), 60)
        
        # 格式化时间戳和消息
        timestamp = f"[已用时 {minutes:02d}:{seconds:02d}]"
        
        # 对主要步骤添加醒目的分隔符，使其更突出
        if is_major_step:
            print("\n" + f"🚀 {timestamp} {message}", file=sys.stdout, flush=True)
        else:
            print(f"   {timestamp} {message}", file=sys.stdout, flush=True)

    def success(self, message: str):
        """
        打印最终的成功信息。
        """
        if not self.enabled:
            return

        elapsed_seconds = time.monotonic() - self.start_time
        minutes, seconds = divmod(int(elapsed_seconds), 60)
        timestamp = f"[总用时 {minutes:02d}:{seconds:02d}]"

        print("\n" + "✅" * 3, file=sys.stdout, flush=True)
        print(f"🎉 {timestamp} {message}", file=sys.stdout, flush=True)
        print("✅" * 3 + "\n", file=sys.stdout, flush=True) 