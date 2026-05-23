import logging

from rich.logging import RichHandler
from rich.traceback import install

install()


class KarnageLogger(logging.Logger):
    def success(self, message, *args, **kwargs):
        super().info(
            f"[bold green]SUCCESS:[/bold green] {message}",
            *args,
            **kwargs
        )

    def failure(self, message, *args, **kwargs):
        super().error(
            f"[bold red]FAILURE:[/bold red] {message}",
            *args,
            **kwargs
        )

    def warning(self, message, *args, **kwargs):
        super().warning(
            f"[bold yellow]WARNING:[/bold yellow] {message}",
            *args,
            **kwargs
        )

    def debug(self, message, *args, **kwargs):
        super().debug(
            f"[dim]DEBUG:[/dim] {message}",
            *args,
            **kwargs
        )
    
    def critical(self, message, *args, **kwargs):
        super().critical(
            f"[bold white on red]CRITICAL:[/bold white on red] {message}",
            *args,
            **kwargs
        )


logging.setLoggerClass(KarnageLogger)

logger = logging.getLogger("KARNAGE")

logger.setLevel(logging.DEBUG)
logger.propagate = False

if not logger.handlers:

    rich_handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_path=False,
        show_time=True,
        show_level=False,
        tracebacks_show_locals=True,
    )

    formatter = logging.Formatter(
        "[bold turquoise4][KARNAGE][/bold turquoise4] %(message)s")

    rich_handler.setFormatter(formatter)

    logger.addHandler(rich_handler)