import sys, http.server, threading, functools
from playwright.sync_api import sync_playwright
root, anchor, out = sys.argv[1], sys.argv[2], sys.argv[3]
h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=root)
srv = http.server.ThreadingHTTPServer(('127.0.0.1', 8765), h)
threading.Thread(target=srv.serve_forever, daemon=True).start()
with sync_playwright() as p:
    b = p.chromium.launch(executable_path='/opt/pw-browsers/chromium-1194/chrome-linux/chrome', args=['--no-sandbox'])
    pg = b.new_page(viewport={'width': 430, 'height': 900})
    pg.goto(f'http://127.0.0.1:8765/api/daf/{anchor}')
    pg.wait_for_timeout(800)
    pg.screenshot(path=out, full_page=False)
    b.close()
srv.shutdown()
