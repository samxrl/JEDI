# Modified by the JEDI CAST adapter: rich console output is optional.
try:
    from rich.console import Console
    console = Console(quiet=False)
except ImportError:
    class _Console:
        def print(self, message, *args, **kwargs):
            print(message)

    console = _Console()
