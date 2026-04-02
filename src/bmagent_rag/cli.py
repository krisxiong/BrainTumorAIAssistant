from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .config import build_config, load_env_file
from .sync import sync_knowledge_base


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Build a local brain tumor MRI knowledge base index.'
    )
    parser.add_argument('--source-dir', type=Path, default=None, help='Local document folder to scan recursively.')
    parser.add_argument('--state-dir', type=Path, default=None, help='Folder for the local manifest and index files.')
    parser.add_argument('--manifest-path', type=Path, default=None, help='Override the manifest file path.')
    parser.add_argument('--index-path', type=Path, default=None, help='Override the local index file path.')
    parser.add_argument('--knowledge-base-name', default=None, help='Human-readable local knowledge base name.')
    parser.add_argument('--knowledge-base-id', default=None, help='Optional fixed local knowledge base id.')
    parser.add_argument('--allowed-extensions', default=None, help='Comma-separated list of allowed file extensions.')
    parser.add_argument('--chunk-size-chars', type=int, default=None, help='Chunk size in characters.')
    parser.add_argument('--chunk-overlap-chars', type=int, default=None, help='Chunk overlap in characters.')
    parser.add_argument('--dry-run', action='store_true', help='Scan and report what would be indexed without writing files.')
    parser.add_argument('--env-file', type=Path, default=Path('.env'), help='Optional .env file to load before resolving environment variables.')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.env_file:
        load_env_file(args.env_file)

    allowed_extensions = None
    if args.allowed_extensions:
        allowed_extensions = [item.strip() for item in args.allowed_extensions.split(',') if item.strip()]

    config = build_config(
        source_dir=args.source_dir,
        state_dir=args.state_dir,
        manifest_path=args.manifest_path,
        index_path=args.index_path,
        knowledge_base_name=args.knowledge_base_name,
        knowledge_base_id=args.knowledge_base_id,
        allowed_extensions=allowed_extensions,
        chunk_size_chars=args.chunk_size_chars,
        chunk_overlap_chars=args.chunk_overlap_chars,
        dry_run=args.dry_run,
    )
    summary = sync_knowledge_base(config)

    print(f'Source root: {summary.source_root}')
    print(f'Manifest: {summary.manifest_path}')
    print(f'Index: {summary.index_path}')
    print(f'Knowledge base ID: {summary.knowledge_base_id}')
    print(f'Total files: {summary.total_files}')
    print(f'New or changed: {summary.new_or_changed_files}')
    print(f'Skipped: {summary.skipped_files}')
    print(f'Indexed: {summary.indexed_files}')
    print(f'Failed: {summary.failed_files}')
    print(f'Chunks: {summary.chunk_count}')
    print(f'Dry run: {summary.dry_run}')
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())