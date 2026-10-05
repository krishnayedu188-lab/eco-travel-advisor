"""
Small web server used in deployment: serves the chat page and forwards chat
messages to the Rasa server. This means only ONE public port is needed
(HuggingFace Spaces: 7860) and the Rasa and action servers stay private.
"""
import http.server
import os
import urllib.error
import urllib.request

PORT = int(os.getenv("PORT", "7860"))
RASA_URL = "http://localhost:5005/webhooks/rest/webhook"
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
MAX_BODY = 10_000  # bytes - chat messages are tiny; reject anything larger


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=FRONTEND_DIR, **kwargs)

    def do_POST(self):
        if self.path != "/webhooks/rest/webhook":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_BODY:
            self.send_error(413)
            return
        request = urllib.request.Request(
            RASA_URL, data=self.rfile.read(length), headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data, status = response.read(), response.status
        except (urllib.error.URLError, TimeoutError):
            self.send_error(503, "The assistant is starting up - please try again in a moment")
            return
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()