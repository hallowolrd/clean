from __future__ import annotations

import re
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO


class TeeStream:
    """
    双写输出流。

    作用：
    把写入 stdout / stderr 的内容同时写到：
    1. 原始控制台
    2. 日志文件

    额外能力：
    可以过滤 tqdm 这类动态进度条，避免 train.log 被进度条污染。
    """

    def __init__(
        self,
        console_stream: TextIO,
        log_file: TextIO,
        *,
        filter_progress: bool = False,
    ) -> None:
        self.console_stream = console_stream
        self.log_file = log_file
        self.filter_progress = bool(filter_progress)
        self.lock = threading.Lock()

    def _should_write_to_log(self, text: str) -> bool:
        """
        判断当前输出是否应该写入日志文件。

        tqdm / rich / 部分终端动态刷新通常会包含：
        1. \\r：回到行首刷新进度条
        2. ANSI 控制符：例如 \\x1b[...m
        3. tqdm 常见片段：%|、it/s、s/it 等

        这些内容适合显示在控制台，但不适合写入 train.log。
        """
        if not self.filter_progress:
            return True

        if not text:
            return False

        # tqdm 动态刷新最常见特征：使用 \r 回到行首重绘。
        if "\r" in text:
            return False

        # 过滤 ANSI 控制符，避免颜色、清屏、光标移动等控制字符进入日志。
        if "\x1b[" in text:
            return False

        # 过滤常见 tqdm 进度条文本。
        progress_patterns = [
            r"\d+%\|",      # 例如： 34%|
            r"\|\s*\d+/",   # 例如： | 34/100
            r"it/s",        # 例如： 2.81it/s
            r"s/it",        # 例如： 1.23s/it
            r"\[\d{2}:\d{2}",  # 例如： [00:12<00:23
        ]

        for pattern in progress_patterns:
            if re.search(pattern, text):
                return False

        return True

    def write(self, text: str) -> int:
        """
        写入控制台，并在必要时写入日志文件。

        注意：
        控制台永远保留原始输出；
        只有日志文件会过滤进度条。
        """
        with self.lock:
            self.console_stream.write(text)
            self.console_stream.flush()

            if self._should_write_to_log(text):
                self.log_file.write(text)
                self.log_file.flush()

        return len(text)

    def flush(self) -> None:
        """
        同时刷新控制台和日志文件。
        """
        with self.lock:
            self.console_stream.flush()
            self.log_file.flush()

    def isatty(self) -> bool:
        """
        保留控制台 TTY 判断能力。

        这对 tqdm / 某些终端输出工具有用。
        """
        return self.console_stream.isatty()

    @property
    def encoding(self) -> str:
        """
        返回原始控制台编码。
        """
        return getattr(self.console_stream, "encoding", "utf-8")


@contextmanager
def tee_output_to_file(
    log_path: str | Path,
    *,
    filter_stderr_progress: bool = True,
) -> Iterator[None]:
    """
    把 stdout 和 stderr 同时写入日志文件。

    用法：
        with tee_output_to_file("outputs/run/train.log"):
            print("hello")

    效果：
    1. 控制台能看到 hello
    2. train.log 里也会保存 hello

    额外说明：
    - stdout 默认不过滤，普通 print 会进入 train.log。
    - stderr 默认过滤进度条，因为 tqdm 通常写 stderr。
    - traceback / 报错信息一般不包含 tqdm 特征，所以仍会进入 train.log。
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    old_stdout = sys.stdout
    old_stderr = sys.stderr

    with log_path.open(
        "a",
        encoding="utf-8",
        buffering=1,
    ) as log_file:
        sys.stdout = TeeStream(
            console_stream=old_stdout,
            log_file=log_file,
            filter_progress=False,
        )

        sys.stderr = TeeStream(
            console_stream=old_stderr,
            log_file=log_file,
            filter_progress=filter_stderr_progress,
        )

        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr