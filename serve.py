#!/usr/bin/env python3
"""Local image classifier. Run with .venv/bin/python serve.py."""
import argparse
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import torch
from PIL import Image
from image_classifier.checkpoint import load_model
from image_classifier.data import CLASSES, load_rgb, transforms_for
from image_classifier.model import resolve_device

ROOT = Path(__file__).resolve().parent
MAX_UPLOAD = 20 * 1024 * 1024

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'runs/20261021-204523/final.pt')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--open', action='store_true', help='Open the page when the model is ready')
    args = parser.parse_args()
    device = resolve_device(args.device)
    model, state = load_model(args.checkpoint, device)
    if state.get('status') != 'refitted':
        raise ValueError('Choose a final refitted checkpoint.')
    del state
    transform = transforms_for(False)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def respond(self, status, body, content_type='application/json'):
            if isinstance(body, dict):
                body = json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(body)

        def local_request(self):
            allowed = {f'127.0.0.1:{args.port}', f'localhost:{args.port}'}
            origin = self.headers.get('Origin')
            return self.headers.get('Host') in allowed and (origin is None or origin in {'http://'+v for v in allowed})

        def do_GET(self):
            if not self.local_request():
                return self.respond(403, {'error': 'Local access only.'})
            routes = {'/': ('index.html','text/html; charset=utf-8'),
                      '/app.js': ('app.js','text/javascript; charset=utf-8')}
            if self.path == '/health':
                return self.respond(200, {'ready': True})
            if self.path not in routes:
                return self.respond(404, {'error': 'Not found.'})
            name, mime = routes[self.path]
            self.respond(200, (ROOT/'web'/name).read_bytes(), mime)

        def do_POST(self):
            if not self.local_request():
                return self.respond(403, {'error': 'Local access only.'})
            if self.path != '/predict':
                return self.respond(404, {'error': 'Not found.'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
            except ValueError:
                return self.respond(400, {'error': 'Invalid upload size.'})
            if not 0 < size <= MAX_UPLOAD:
                self.close_connection = True
                return self.respond(413, {'error': 'Choose an image smaller than 20 MB.'})
            if not lock.acquire(blocking=False):
                self.close_connection = True
                return self.respond(429, {'error': 'An image is being analyzed. Try again shortly.'})
            try:
                raw = self.rfile.read(size)
                if len(raw) != size:
                    return self.respond(400, {'error': 'Upload was interrupted. Please try again.'})
                try:
                    with Image.open(io.BytesIO(raw)) as probe:
                        if probe.width * probe.height > 25_000_000:
                            return self.respond(413, {'error': 'Choose an image under 25 megapixels.'})
                    image = load_rgb(io.BytesIO(raw))
                except Exception:
                    return self.respond(400, {'error': 'Could not read this image. Try a PNG, JPEG, or WebP.'})
                with torch.inference_mode():
                    scores = model(transform(image).unsqueeze(0).to(device)).softmax(1)[0].cpu().tolist()
                label = list(CLASSES)[max(range(3), key=scores.__getitem__)]
                self.respond(200, {'label': label, 'scores': dict(zip(CLASSES, scores))})
            except Exception as error:
                print(f'Prediction failed: {error}', flush=True)
                self.respond(500, {'error': 'Could not analyze the image. Please try again.'})
            finally:
                lock.release()

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.daemon_threads = True
    print(f'Image classifier ready at http://localhost:{args.port} ({device})', flush=True)
    if args.open:
        import webbrowser
        threading.Timer(.3, lambda: webbrowser.open(f'http://localhost:{args.port}')).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__ == '__main__':
    main()
