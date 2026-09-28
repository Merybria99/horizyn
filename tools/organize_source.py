"""Audit/organize maintained source; historical artifacts are never traversed.

Use --apply only for the initial migration; later --check verifies local imports.
The migration manifest records every original path, archive path, and SHA256.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'archive/20260928'
SCRIPT_ROOTS = {
    'train_protein_pooling', 'run_unified_retrieval_benchmark',
    'extract_prott5_residue_embeddings', 'extract_reaction_t5v2_embeddings',
    'extract_unimol2_reaction_embeddings', 'extract_chiro_reaction_embeddings',
    'build_annotation_negative_pools', 'prepare_paper_benchmark_data',
    'train_sleec_stage1', 'generalization_clipzyme_f3_screen',
    'generalization_multiview_calibration', 'generalization_reactzyme_architecture_phase2',
}


def sources():
    paths = list((ROOT/'horizyn').rglob('*.py')) + list((ROOT/'scripts').glob('*.py'))
    paths += list((ROOT/'wet_lab').glob('*.py'))
    return {str(p.relative_to(ROOT)): p for p in paths if '__pycache__' not in p.parts}


def module_map(files):
    modules = {}
    for key, path in files.items():
        module = key[:-3].replace('/', '.').removesuffix('.__init__')
        modules[module] = key
        if key.startswith('scripts/'):
            modules[path.stem] = key
    return modules


def dependencies(path, key, modules):
    try:
        tree = ast.parse(path.read_text())
    except (OSError, UnicodeError, SyntaxError):
        return set()
    package = key[:-3].replace('/', '.').rsplit('.', 1)[0]
    if key.endswith('/__init__.py'):
        package = key[:-12].replace('/', '.')
    found = set()
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ''
            if node.level:
                base = '.'.join(package.split('.')[:len(package.split('.'))-node.level+1]) + ('.'+base if base else '')
            names = [base] + [base+'.'+a.name for a in node.names]
        for name in names:
            if name in modules:
                found.add(modules[name])
            for count in range(1, len(name.split('.'))):
                parent = modules.get('.'.join(name.split('.')[:count]))
                if parent and parent.endswith('__init__.py'):
                    found.add(parent)
    return found


def plan():
    files = sources()
    modules = module_map(files)
    roots = [key for key in files if key.startswith('horizyn/pipelines/')]
    roots += ['horizyn/__init__.py']
    roots += ['scripts/'+name+'.py' for name in SCRIPT_ROOTS if 'scripts/'+name+'.py' in files]
    for path in (ROOT.parent/'case_studies').glob('*.py'):
        roots.extend(dependencies(path, str(path), modules))
    active, queue = set(), list(roots)
    while queue:
        key = queue.pop()
        if key in active or key not in files:
            continue
        active.add(key)
        queue.extend(dependencies(files[key], key, modules)-active)
    return files, active


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    files, active = plan()
    if args.check:
        errors = []
        modules = module_map(files)
        for key in active:
            tree = ast.parse(files[key].read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('horizyn.'):
                    if node.module not in modules:
                        errors.append(f'{key}:{node.lineno}: missing {node.module}')
        if errors:
            raise SystemExit('\n'.join(errors))
        print(f'All local imports resolve across {len(active)} maintained dependency files')
        return
    moves = set(files)-active
    # Obsolete shell launchers are intentionally not maintained entry points.
    moves.update(str(p.relative_to(ROOT)) for p in (ROOT/'scripts').glob('*.sh'))
    # Keep only clean pipeline configs and official benchmark definitions.
    moves.update(str(p.relative_to(ROOT)) for p in (ROOT/'configs').rglob('*')
                 if p.is_file() and p.suffix in {'.yaml', '.yml'}
                 and 'pipelines' not in p.parts and 'benchmarks' not in p.parts)
    moves.update(str(p.relative_to(ROOT)) for p in (ROOT/'configs/benchmarks').rglob('*')
                 if p.is_file() and p.suffix in {'.yaml', '.yml'}
                 and p.name not in {'enzyme_retrieval_unified.yaml'})
    print('Maintained scripts:', *sorted(k for k in active if k.startswith('scripts/')), sep='\n')
    print(f'Maintained source: {len(active)}; files to archive: {len(moves)}')
    records = []
    for key in sorted(moves):
        path = ROOT/key
        record = {'original': key, 'archive': str((ARCHIVE/key).relative_to(ROOT))}
        try:
            record['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            record['unreadable'] = str(error)
        if args.apply:
            destination = ARCHIVE/key
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise ValueError(f'Archive collision: {destination}')
            path.rename(destination)
        records.append(record)
    if args.apply:
        (ARCHIVE/'migration.json').write_text(json.dumps(records, indent=2)+'\n')
        (ARCHIVE/'active_sources.json').write_text(json.dumps(sorted(active), indent=2)+'\n')


if __name__ == '__main__':
    main()
