"""
Serveur de Podcast Local.

Génère un flux RSS pour les livres audio générés et les sert via HTTP.
Permet d'écouter les livres sur un téléphone via une appli de podcast.
"""
import os
import socket
import threading
import datetime
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import List, Dict
import html
import mimetypes

# Imports conditionnels
try:
    import qrcode
    from io import BytesIO
    import base64
    QRCODE_AVAILABLE = True
except ImportError:
    QRCODE_AVAILABLE = False


class PodcastFeedGenerator:
    """Génère le flux RSS du podcast."""

    def __init__(self, base_url: str, title: str = "AudioReader Library"):
        self.base_url = base_url.rstrip('/')
        self.title = title
        self.items: List[Dict] = []

    def scan_directory(self, root_dir: Path):
        """Scanne le dossier pour trouver les fichiers audio."""
        root_dir = Path(root_dir)
        
        # Extensions supportées
        extensions = {'.mp3', '.m4b', '.m4a'}
        
        for path in root_dir.rglob('*'):
            if path.suffix.lower() in extensions:
                # Créer l'item
                rel_path = path.relative_to(root_dir)
                url = f"{self.base_url}/{rel_path.as_posix()}"
                
                size = path.stat().st_size
                mtime = datetime.datetime.fromtimestamp(path.stat().st_mtime)
                
                # Essayer de deviner le titre depuis le dossier ou le fichier
                if path.parent != root_dir:
                    title = f"{path.parent.name} - {path.stem}"
                else:
                    title = path.stem
                    
                mime_type = mimetypes.guess_type(path)[0] or 'audio/mpeg'
                
                self.items.append({
                    'title': title,
                    'url': url,
                    'size': size,
                    'type': mime_type,
                    'date': mtime,
                    'filename': path.name
                })
        
        # Trier par date décroissante
        self.items.sort(key=lambda x: x['date'], reverse=True)

    def generate_xml(self) -> str:
        """Génère le XML du flux RSS."""
        xml = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">',
            '<channel>',
            f'<title>{html.escape(self.title)}</title>',
            f'<link>{self.base_url}</link>',
            '<description>Bibliothèque AudioReader locale</description>',
            '<language>fr</language>',
        ]
        
        for item in self.items:
            date_str = item['date'].strftime("%a, %d %b %Y %H:%M:%S +0000")
            xml.append('<item>')
            xml.append(f'<title>{html.escape(item["title"])}</title>')
            xml.append(f'<enclosure url="{item["url"]}" length="{item["size"]}" type="{item["type"]}" />')
            xml.append(f'<guid>{item["url"]}</guid>')
            xml.append(f'<pubDate>{date_str}</pubDate>')
            xml.append('</item>')
            
        xml.append('</channel>')
        xml.append('</rss>')
        
        return '\n'.join(xml)


class PodcastServer:
    """Serveur HTTP pour le podcast."""

    def __init__(self, root_dir: Path, port: int = 8080):
        self.root_dir = Path(root_dir)
        self.port = port
        self.server = None
        self.thread = None
        self.is_running = False
        
        # Déterminer l'IP locale
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            self.ip = s.getsockname()[0]
            s.close()
        except:
            self.ip = "127.0.0.1"
            
        self.base_url = f"http://{self.ip}:{self.port}"
        self.feed_url = f"{self.base_url}/feed.xml"

    def start(self):
        """Démarre le serveur dans un thread."""
        if self.is_running:
            return

        # Générer le feed initial
        self.refresh_feed()
        
        # Handler personnalisé pour servir le dossier root_dir
        root = self.root_dir
        
        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(root), **kwargs)
                
            def log_message(self, format, *args):
                pass # Silence logs
                
        self.server = HTTPServer(('0.0.0.0', self.port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.daemon = True
        self.thread.start()
        self.is_running = True
        print(f"Serveur Podcast démarré sur {self.feed_url}")

    def stop(self):
        """Arrête le serveur."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
            self.is_running = False

    def refresh_feed(self):
        """Met à jour le fichier feed.xml."""
        generator = PodcastFeedGenerator(self.base_url)
        generator.scan_directory(self.root_dir)
        xml = generator.generate_xml()
        
        with open(self.root_dir / "feed.xml", "w", encoding="utf-8") as f:
            f.write(xml)

    def get_qr_code(self) -> str:
        """Retourne le QR code du flux en base64 (HTML img tag)."""
        if not QRCODE_AVAILABLE:
            return "<p>Module qrcode non installé (pip install qrcode)</p>"
            
        qr = qrcode.QRCode(box_size=10, border=4)
        qr.add_data(self.feed_url)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f'<img src="data:image/png;base64,{img_str}" alt="QR Code" width="200" />'


# Singleton global
_server_instance = None

def get_server(root_dir: str = "output", port: int = 8080) -> PodcastServer:
    global _server_instance
    if _server_instance is None:
        _server_instance = PodcastServer(Path(root_dir), port)
    return _server_instance
