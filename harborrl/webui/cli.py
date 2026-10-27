"""Start a single-user loopback console, optionally opening the browser."""
import argparse
import contextlib
import os
import signal
import webbrowser
from pathlib import Path


@contextlib.contextmanager
def workspace_lock(workspace):
    workspace.mkdir(parents=True, exist_ok=True)
    with (workspace / ".console.lock").open("a+b") as stream:
        try:
            if os.name == "posix":
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                import msvcrt
                stream.seek(0)
                if stream.read(1) == b"":
                    stream.write(b"0")
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise ValueError("Another console owns this workspace") from exc
        yield


def main(argv):
    parser = argparse.ArgumentParser(prog="harborrl web", description=__doc__)
    parser.add_argument("--workspace", default="./harborrl-web", help="Configurations and job logs directory")
    parser.add_argument("--port", type=int, default=0, help="Loopback port; 0 selects a free port")
    parser.add_argument("--no-browser", action="store_true", help="Print the URL without opening a browser")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("port must be in 0..65535")
    from .server import Console, make_server

    workspace = Path(args.workspace).expanduser().resolve()
    try:
        with workspace_lock(workspace):
            console = Console(workspace)
            server = make_server(console, args.port)
            url = f"http://127.0.0.1:{server.server_port}/#token={console.token}"
            print(f"HarborRL console: {url}\nWorkspace: {workspace}\nKeep this terminal open. Ctrl+C stops the console and requests active job cleanup.", flush=True)
            if not args.no_browser:
                try:
                    webbrowser.open(url)
                except webbrowser.Error:
                    print("Browser unavailable; open the printed URL.", flush=True)
            previous = signal.getsignal(signal.SIGTERM)
            def terminate(_signum, _frame):
                raise KeyboardInterrupt
            signal.signal(signal.SIGTERM, terminate)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
                console.jobs.close()
                signal.signal(signal.SIGTERM, previous)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0
