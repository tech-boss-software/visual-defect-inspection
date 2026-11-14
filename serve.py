"""Single-page demo (standard library): drop a part image, get the anomaly heatmap and a calibrated verdict. Local use only."""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from anomaly import MemoryBank
from render import overlay, side_by_side

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Visual defect inspection</title>
<style>body{font-family:system-ui,sans-serif;margin:24px;color:#1f2937;background:#f8fafc}#drop{border:2px dashed #94a3b8;padding:40px;text-align:center;background:#fff;border-radius:8px}
#drop.over{border-color:#2563eb;background:#eff6ff}.row{display:flex;gap:20px;flex-wrap:wrap;margin-top:20px}img{max-width:560px;border:1px solid #e2e8f0}
pre{background:#fff;border:1px solid #e2e8f0;padding:12px;max-width:520px;overflow:auto;font-size:12px}.ok{color:#16a34a;font-weight:700;font-size:22px}.bad{color:#dc2626;font-weight:700;font-size:22px}</style></head>
<body><h1>Visual defect inspection</h1><p>Bank: <code>__BANK__</code> · threshold <code>__THRESHOLD__</code> (calibrated at __FA__ false alarms on held-out good parts). Drop a part image; everything runs on this machine.</p>
<div id="drop">Drop an image here or <input type="file" id="file"></div><div id="status"></div>
<div class="row"><div id="image"></div><div id="result"></div></div>
<script>
const drop=document.getElementById('drop');const fileInput=document.getElementById('file');
['dragenter','dragover'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.add('over')}));
['dragleave','drop'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.remove('over')}));
drop.addEventListener('drop',ev=>send(ev.dataTransfer.files[0]));fileInput.addEventListener('change',()=>send(fileInput.files[0]));
async function send(file){if(!file)return;document.getElementById('status').textContent='Scoring '+file.name+' ...';
const res=await fetch('/score?name='+encodeURIComponent(file.name),{method:'POST',body:file});const data=await res.json();
document.getElementById('status').textContent='';if(data.error){document.getElementById('result').innerHTML='<p class="bad">'+data.error+'</p>';return}
document.getElementById('image').innerHTML='<img src="data:image/png;base64,'+data.panel_png_base64+'">';
const v=data.verdict;document.getElementById('result').innerHTML=(v.defect?'<div class="bad">DEFECT</div>':'<div class="ok">OK</div>')+'<pre>'+JSON.stringify(data.verdict,null,2)+'</pre>'}
</script></body></html>"""


def make_handler(bank, extractor, sigma=4.0, bank_name="bank"):
    page = PAGE.replace("__BANK__", bank_name).replace("__THRESHOLD__", f"{bank.threshold:.4f}").replace(
        "__FA__", f"{bank.calibration['target_false_alarm']:.0%}" if bank.calibration else "?")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_GET(self):
            if self.path.split("?")[0] != "/":
                self.send_error(404)
                return
            body = page.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if not self.path.startswith("/score"):
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 30 * 1024 * 1024:
                self._json({"error": "empty or oversized upload"}, 400)
                return
            payload = self.rfile.read(length)
            match = re.search(r"name=([^&]+)", self.path)
            try:
                result = score_upload(payload, match.group(1) if match else "upload", bank, extractor, sigma)
            except Exception as error:  # the page shows the message; the server keeps running
                self._json({"error": f"{type(error).__name__}: {error}"}, 400)
                return
            self._json(result)

        def _json(self, payload, status=200):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def score_upload(payload, name, bank, extractor, sigma=4.0):
    from urllib.parse import unquote

    from PIL import Image

    image = Image.open(io.BytesIO(payload)).convert("RGB")
    started = time.perf_counter()
    score, heat, _ = bank.score(extractor.extract(image), (image.height, image.width), sigma)
    verdict = bank.verdict(score)
    verdict.update({"file": unquote(name), "score_ms": round((time.perf_counter() - started) * 1000, 1),
                    "hottest_xy": [int(v) for v in np.unravel_index(int(np.argmax(heat)), heat.shape)[::-1]]})
    low = bank.calibration["good_score_median"] if bank.calibration else 0.0
    high = max(bank.threshold, score) * 1.05
    panel = side_by_side([image, overlay(image, heat, low, high)], [unquote(name), f"score {score:.3f} vs threshold {bank.threshold:.3f}"])
    buffer = io.BytesIO()
    panel.save(buffer, "PNG")
    return {"verdict": verdict, "panel_png_base64": base64.b64encode(buffer.getvalue()).decode("ascii")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, help="folder written by inspect_anomaly.py fit + calibrate")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--sigma", type=float, default=4.0)
    args = parser.parse_args(argv)
    bank = MemoryBank.load(args.bank)
    if bank.threshold is None:
        raise SystemExit("bank is not calibrated; run inspect_anomaly.py calibrate first")
    from features import ResNetPatchFeatures

    extractor = ResNetPatchFeatures(input_size=bank.feature_info.get("input_size", 224), threads=args.threads, cache_dir=args.cache_dir)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(bank, extractor, args.sigma, args.bank))
    print(f"serving on http://{args.host}:{args.port}/ (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
