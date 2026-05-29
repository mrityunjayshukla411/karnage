import logging
import time

from rich.logging import RichHandler
from rich.table import Table
from rich.traceback import install

install()


class KarnageFormatter(logging.Formatter):
    def __init__(self, fmt=None):
        super().__init__(fmt)
        self.last_time = time.perf_counter()

    def format(self, record):
        now = time.perf_counter()
        elapsed = now - self.last_time
        self.last_time = now

        record.elapsed = f"~{elapsed:.2f}s"

        return super().format(record)


class KarnageRichHandler(RichHandler):
    def render_message(self, record, message):
        table = Table.grid(expand=True)
        table.add_column(ratio=1)
        table.add_column(justify="right", width=10)

        table.add_row(message, f"[turquoise4]{record.elapsed}[/turquoise4]")

        return table


class KarnageLogger(logging.Logger):
    def success(self, message, *args, **kwargs):
        super().info(f"[bold green]SUCCESS:[/bold green] {message}", *args, **kwargs)

    def failure(self, message, *args, **kwargs):
        super().error(f"[bold red]FAILURE:[/bold red] {message}", *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        super().warning(
            f"[bold yellow]WARNING:[/bold yellow] {message}", *args, **kwargs
        )

    def debug(self, message, *args, **kwargs):
        super().debug(f"[dim]DEBUG:[/dim] {message}", *args, **kwargs)

    def critical(self, message, *args, **kwargs):
        super().critical(
            f"[bold white on red]CRITICAL:[/bold white on red] {message}",
            *args,
            **kwargs,
        )

    def info(self, message, *args, **kwargs):
        super().info(f"[cyan2]INFO:[/cyan2] {message}", *args, **kwargs)


logging.setLoggerClass(KarnageLogger)

logger = logging.getLogger("KARNAGE")

logger.setLevel(logging.DEBUG)
logger.propagate = False

if not logger.handlers:

    rich_handler = KarnageRichHandler(
        rich_tracebacks=True,
        markup=True,
        show_path=False,
        show_time=False,
        show_level=False,
        tracebacks_show_locals=True,
    )

    formatter = KarnageFormatter("[bold cyan][KARNAGE][/bold cyan] %(message)s")

    rich_handler.setFormatter(formatter)
    logger.addHandler(rich_handler)
