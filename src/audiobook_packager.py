"""
Audiobook Packager - Crée des fichiers M4B avec chapitres et métadonnées.

Fonctionnalités:
- Fusion de fichiers WAV en un seul audiobook
- Chapitrage automatique avec timestamps
- Métadonnées (titre, auteur, narrateur, couverture)
- Export M4B compatible iTunes/Apple Books/VLC
"""

import os
import subprocess
import json
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import re


@dataclass
class Chapter:
    """Représente un chapitre de l'audiobook."""
    title: str
    start_ms: int = 0
    end_ms: int = 0
    audio_file: Optional[str] = None


@dataclass
class AudiobookMetadata:
    """Métadonnées de l'audiobook."""
    title: str = "Audiobook"
    author: str = "Unknown Author"
    narrator: str = "AI Voice"
    year: str = ""
    genre: str = "Audiobook"
    description: str = ""
    cover_image: Optional[str] = None


@dataclass
class AudiobookProject:
    """Projet d'audiobook complet."""
    metadata: AudiobookMetadata
    chapters: List[Chapter] = field(default_factory=list)


def get_audio_duration_ms(audio_file: str) -> int:
    """Récupère la durée d'un fichier audio en millisecondes."""
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_format', audio_file
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe error: {result.stderr}")

    data = json.loads(result.stdout)
    duration_s = float(data['format']['duration'])
    return int(duration_s * 1000)


def format_ffmpeg_time(ms: int) -> str:
    """Convertit les millisecondes en format HH:MM:SS.mmm"""
    hours = ms // 3600000
    minutes = (ms % 3600000) // 60000
    seconds = (ms % 60000) // 1000
    millis = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def create_chapters_file(chapters: List[Chapter], output_path: str) -> str:
    """Crée un fichier de chapitres au format FFMETADATA."""
    lines = [";FFMETADATA1"]

    for i, chapter in enumerate(chapters):
        lines.append("")
        lines.append("[CHAPTER]")
        lines.append("TIMEBASE=1/1000")
        lines.append(f"START={chapter.start_ms}")
        lines.append(f"END={chapter.end_ms}")
        # Escape special characters in title
        safe_title = chapter.title.replace("=", "\\=").replace(";", "\\;").replace("#", "\\#").replace("\\", "\\\\")
        lines.append(f"title={safe_title}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return output_path


def create_concat_file(audio_files: List[str], output_path: str) -> str:
    """Crée un fichier de concaténation pour ffmpeg."""
    lines = []
    for f in audio_files:
        # Utiliser le chemin absolu et escape single quotes
        abs_path = os.path.abspath(f)
        safe_path = abs_path.replace("'", "'\\''")
        lines.append(f"file '{safe_path}'")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return output_path


def package_audiobook(
    audio_files: List[str],
    output_file: str,
    metadata: Optional[AudiobookMetadata] = None,
    chapter_titles: Optional[List[str]] = None,
    bitrate: str = "128k",
    verbose: bool = False
) -> str:
    """
    Crée un audiobook M4B à partir de fichiers audio.

    Args:
        audio_files: Liste des fichiers audio (WAV/MP3) dans l'ordre
        output_file: Chemin du fichier M4B de sortie
        metadata: Métadonnées de l'audiobook
        chapter_titles: Titres des chapitres (optionnel, sinon "Chapitre N")
        bitrate: Bitrate audio (défaut: 128k)
        verbose: Afficher les commandes ffmpeg

    Returns:
        Chemin du fichier M4B créé
    """
    if not audio_files:
        raise ValueError("Aucun fichier audio fourni")

    if metadata is None:
        metadata = AudiobookMetadata()

    # Créer les titres de chapitres si non fournis
    if chapter_titles is None:
        chapter_titles = [f"Chapitre {i+1}" for i in range(len(audio_files))]

    # S'assurer qu'on a le bon nombre de titres
    while len(chapter_titles) < len(audio_files):
        chapter_titles.append(f"Chapitre {len(chapter_titles)+1}")

    # Calculer les durées et créer les chapitres
    chapters = []
    current_ms = 0

    print("Analyse des fichiers audio...")
    for i, audio_file in enumerate(audio_files):
        duration_ms = get_audio_duration_ms(audio_file)
        chapter = Chapter(
            title=chapter_titles[i],
            start_ms=current_ms,
            end_ms=current_ms + duration_ms,
            audio_file=audio_file
        )
        chapters.append(chapter)
        current_ms += duration_ms

        if verbose:
            print(f"  {chapter_titles[i]}: {format_ffmpeg_time(duration_ms)}")

    total_duration = format_ffmpeg_time(current_ms)
    print(f"Durée totale: {total_duration}")

    # Créer les fichiers temporaires
    with tempfile.TemporaryDirectory() as tmpdir:
        concat_file = os.path.join(tmpdir, "concat.txt")
        chapters_file = os.path.join(tmpdir, "chapters.txt")
        temp_audio = os.path.join(tmpdir, "combined.m4a")

        # Créer le fichier de concaténation
        create_concat_file(audio_files, concat_file)

        # Créer le fichier de chapitres
        create_chapters_file(chapters, chapters_file)

        # Étape 1: Concaténer et encoder en AAC
        print("Encodage audio AAC...")
        cmd1 = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', concat_file,
            '-c:a', 'aac', '-b:a', bitrate,
            '-ar', '44100',  # Sample rate standard
            '-ac', '1',      # Mono (plus petit, suffisant pour voix)
            temp_audio
        ]

        if not verbose:
            cmd1.insert(1, '-loglevel')
            cmd1.insert(2, 'warning')

        result = subprocess.run(cmd1, capture_output=not verbose)
        if result.returncode != 0:
            raise RuntimeError(f"Erreur encodage: {result.stderr}")

        # Étape 2: Ajouter les métadonnées et chapitres
        print("Ajout des métadonnées et chapitres...")

        # Construire les métadonnées
        meta_args = [
            '-metadata', f'title={metadata.title}',
            '-metadata', f'artist={metadata.author}',
            '-metadata', f'album={metadata.title}',
            '-metadata', f'composer={metadata.narrator}',
            '-metadata', f'genre={metadata.genre}',
            '-metadata', f'date={metadata.year}',
            '-metadata', f'comment={metadata.description}',
        ]

        cmd2 = [
            'ffmpeg', '-y',
            '-i', temp_audio,
            '-i', chapters_file,
            '-map_metadata', '1',
            '-map_chapters', '1',
            '-c', 'copy',
            *meta_args,
        ]

        # Ajouter la couverture si disponible
        if metadata.cover_image and os.path.exists(metadata.cover_image):
            cmd2 = [
                'ffmpeg', '-y',
                '-i', temp_audio,
                '-i', chapters_file,
                '-i', metadata.cover_image,
                '-map', '0:a',
                '-map', '2:v',
                '-map_metadata', '1',
                '-map_chapters', '1',
                '-c:a', 'copy',
                '-c:v', 'mjpeg',
                '-disposition:v', 'attached_pic',
                *meta_args,
            ]

        # Assurer l'extension .m4b
        if not output_file.lower().endswith('.m4b'):
            output_file = output_file.rsplit('.', 1)[0] + '.m4b'

        cmd2.append(output_file)

        if not verbose:
            cmd2.insert(1, '-loglevel')
            cmd2.insert(2, 'warning')

        result = subprocess.run(cmd2, capture_output=not verbose)
        if result.returncode != 0:
            raise RuntimeError(f"Erreur métadonnées: {result.stderr}")

    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Audiobook créé: {output_file} ({file_size:.1f} MB)")

    return output_file


def package_from_directory(
    input_dir: str,
    output_file: str,
    metadata: Optional[AudiobookMetadata] = None,
    pattern: str = "*.wav",
    verbose: bool = False
) -> str:
    """
    Crée un audiobook à partir d'un répertoire de fichiers audio.

    Args:
        input_dir: Répertoire contenant les fichiers audio
        output_file: Fichier M4B de sortie
        metadata: Métadonnées
        pattern: Pattern glob pour les fichiers (défaut: *.wav)
        verbose: Mode verbeux
    """
    from glob import glob

    # Trouver les fichiers audio
    audio_files = sorted(glob(os.path.join(input_dir, pattern)))

    if not audio_files:
        raise ValueError(f"Aucun fichier trouvé avec le pattern {pattern} dans {input_dir}")

    # Extraire les titres des noms de fichiers
    chapter_titles = []
    for f in audio_files:
        name = Path(f).stem
        # Nettoyer le nom (enlever numéros de préfixe, underscores, etc.)
        clean = re.sub(r'^[\d_\-\.]+', '', name)
        clean = clean.replace('_', ' ').replace('-', ' ').strip()
        if not clean:
            clean = name
        chapter_titles.append(clean)

    print(f"Trouvé {len(audio_files)} fichiers audio")

    return package_audiobook(
        audio_files=audio_files,
        output_file=output_file,
        metadata=metadata,
        chapter_titles=chapter_titles,
        verbose=verbose
    )


# CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Crée un audiobook M4B avec chapitres",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Depuis un répertoire
  python audiobook_packager.py -i output_livre/ -o mon_livre.m4b -t "Mon Livre" -a "Auteur"

  # Avec couverture
  python audiobook_packager.py -i output/ -o livre.m4b -t "Titre" --cover cover.jpg
        """
    )

    parser.add_argument('-i', '--input', required=True,
                        help='Répertoire contenant les fichiers audio')
    parser.add_argument('-o', '--output', required=True,
                        help='Fichier M4B de sortie')
    parser.add_argument('-t', '--title', default='Audiobook',
                        help='Titre du livre')
    parser.add_argument('-a', '--author', default='Unknown',
                        help='Auteur')
    parser.add_argument('-n', '--narrator', default='AI Voice',
                        help='Narrateur')
    parser.add_argument('--cover', help='Image de couverture (JPG/PNG)')
    parser.add_argument('--year', default='', help='Année de publication')
    parser.add_argument('--description', default='', help='Description')
    parser.add_argument('-p', '--pattern', default='*.wav',
                        help='Pattern pour les fichiers audio (défaut: *.wav)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Mode verbeux')

    args = parser.parse_args()

    metadata = AudiobookMetadata(
        title=args.title,
        author=args.author,
        narrator=args.narrator,
        year=args.year,
        description=args.description,
        cover_image=args.cover
    )

    try:
        output = package_from_directory(
            input_dir=args.input,
            output_file=args.output,
            metadata=metadata,
            pattern=args.pattern,
            verbose=args.verbose
        )
        print(f"\n✓ Audiobook créé avec succès: {output}")
    except Exception as e:
        print(f"\n✗ Erreur: {e}")
        exit(1)
