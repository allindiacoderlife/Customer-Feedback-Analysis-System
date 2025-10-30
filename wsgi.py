"""
Production WSGI Server Configuration
Uses Waitress for Windows or Gunicorn for Linux/Mac
"""
import os
from src.app import app

if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Check if running on Windows
    if os.name == 'nt':  # Windows
        print("🚀 Starting Waitress server (Windows)...")
        from waitress import serve
        serve(app, host='0.0.0.0', port=port, threads=4)
    else:  # Linux/Mac
        print("🚀 Starting Gunicorn server (Linux/Mac)...")
        import gunicorn.app.base
        
        class StandaloneApplication(gunicorn.app.base.BaseApplication):
            def __init__(self, app, options=None):
                self.options = options or {}
                self.application = app
                super().__init__()

            def load_config(self):
                for key, value in self.options.items():
                    self.cfg.set(key.lower(), value)

            def load(self):
                return self.application

        options = {
            'bind': f'0.0.0.0:{port}',
            'workers': 4,
            'timeout': 120,
            'accesslog': '-',
            'errorlog': '-',
        }
        
        StandaloneApplication(app, options).run()
