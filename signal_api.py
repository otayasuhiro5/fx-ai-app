from http.server import HTTPServer, BaseHTTPRequestHandler
import json

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/signal':
            try:
                with open('/root/fx-ai-app/signal.json') as f:
                    data = f.read()
            except:
                data = '{"signal":"WAIT","atr":0.05,"confidence":50,"price":0}'
            self.send_response(200)
            self.send_header('Content-type','application/json')
            self.end_headers()
            self.wfile.write(data.encode())
    def log_message(self,f,*a):
        pass

HTTPServer(('0.0.0.0',8080),Handler).serve_forever()
