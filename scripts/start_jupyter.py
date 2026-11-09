import argparse
import os
from pathlib import Path
import sys

from jupyter_server.serverapp import list_running_servers


root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=8888)
args = parser.parse_args()
for server in list_running_servers():
    if Path(server['root_dir']).resolve() == root:
        print('Jupyter already running at ' + server['url'])
        raise SystemExit(0)
os.execv(sys.executable, [sys.executable, '-m', 'jupyterlab', '--no-browser', '--ip=127.0.0.1',
                         f'--port={args.port}', '--ServerApp.port_retries=0', f'--ServerApp.root_dir={root}'])
