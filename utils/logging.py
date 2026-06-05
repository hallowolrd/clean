from __future__ import annotations

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

    这样普通 print() 不需要改，也能同时进入 train.log。
    """

    def __init__(
        self,
        console_stream: TextIO,
        log_file: TextIO,
    ) -> None:
        self.console_stream = console_stream
        self.log_file = log_file
        self.lock = threading.Lock()

    def write(self, text: str) -> int:
        """
        同时写入控制台和日志文件。
        """
        with self.lock:
            self.console_stream.write(text)
            self.log_file.write(text)

            self.console_stream.flush()
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
def tee_output_to_file(log_path: str | Path) -> Iterator[None]:
    """
    把 stdout 和 stderr 同时写入日志文件。

    用法：
        with tee_output_to_file("outputs/run/train.log"):
            print("hello")

    效果：
        1. 控制台能看到 hello
        2. train.log 里也会保存 hello
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
        )
        sys.stderr = TeeStream(
            console_stream=old_stderr,
            log_file=log_file,
        )

        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr