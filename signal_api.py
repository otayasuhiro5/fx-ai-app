from http.server import HTTPServer, BaseHTTPRequestHandler
import json

class SignalHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/signal':
            try:
                with open('/root/fx-ai-app/signal.json', 'r') as f:
                    data = f.read()
            except:
                data = json.dumps({"signal": "WAIT", "atr": 0.05, "confidence": 50})
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(data.encode())
    def log_message(self, format, *args):
        pass

server = HTTPServer(('0.0.0.0', 5000), SignalHandler)
print("Signal API started")
server.serve_forever()
