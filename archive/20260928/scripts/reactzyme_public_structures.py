#!/usr/bin/env python3
"""Acquire public AlphaFold coordinates for the exact benchmark sequences.

Missing EnzymeCAGE pockets are prioritized; remaining complete structures are
needed by CLIPZyme. Coordinate residue sequences must match the input exactly.
No retrieval checkpoint, activity annotation, or new positive edge is imported.
"""
import argparse
import concurrent.futures
import csv
import hashlib
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/reactzyme_public_baselines_20260921'
AA=dict(zip('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL SEC PYL ASX GLX UNK MSE'.split(),
            'A R N D C Q E G H I L K M F P S T W Y V U O B Z X M'.split()))


def pdb_sequence(contents):
    residues={}
    for line in contents.splitlines():
        if line.startswith(('ATOM  ','HETATM')) and line[12:16].strip()=='CA' and line[16] in (' ','A'):
            key=(line[21],line[22:27])
            residues.setdefault(key,AA.get(line[17:20].strip(),'X'))
    chains={k[0] for k in residues}
    if len(chains)!=1:return None
    return ''.join(residues.values())


class Downloader:
    def __init__(self,rate):self.lock=threading.Lock();self.next=0.;self.rate=rate
    def get(self,url):
        for attempt in range(3):
            with self.lock:
                wait=max(0,self.next-time.monotonic());self.next=max(self.next,time.monotonic())+1/self.rate
            if wait:time.sleep(wait)
            request=urllib.request.Request(url,headers={'User-Agent':'ReactZyme-public-baseline-reproduction/1.0'})
            try:
                with urllib.request.urlopen(request,timeout=30) as response:return response.read()
            except urllib.error.HTTPError as error:
                if error.code not in (429,500,502,503,504) or attempt==2:raise
                time.sleep(min(60,float(error.headers.get('Retry-After',2**(attempt+1)))))
            except (TimeoutError,urllib.error.URLError):
                if attempt==2:raise
                time.sleep(2**(attempt+1))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=8);parser.add_argument('--requests-per-second',type=float,default=12)
    parser.add_argument('--limit',type=int);a=parser.parse_args()
    root=RUN/'assets/alphafold';root.mkdir(parents=True,exist_ok=True)
    cat=json.loads((RUN/'features/catalog.json').read_text());by_sequence={}
    with (ROOT/'data/paper/reactzyme/raw/uniprot_rhea.tsv').open() as f:
        for r in csv.DictReader(f,delimiter='\t'):by_sequence.setdefault(r['Sequence'],set()).add(r['Entry'])
    missing=set(json.loads((RUN/'features/enzymecage_pocket_coverage.json').read_text())['missing_ids'])
    records=sorted(zip(cat['protein_ids'],cat['protein_sequences']),key=lambda x:(x[0] not in missing,x[0]))
    if a.limit:records=records[:a.limit]
    dl=Downloader(a.requests_per_second)
    def download(record):
        pid,sequence=record;folder=root/pid[5:7];folder.mkdir(exist_ok=True)
        receipt=folder/f'{pid}.json';pdb=folder/f'{pid}.pdb'
        if receipt.exists() and pdb.exists():
            meta=json.loads(receipt.read_text())
            if meta.get('sequence_sha256')==hashlib.sha256(sequence.encode()).hexdigest() and meta.get('exact_sequence_match'):return meta
        attempts=[]
        for uid in sorted(by_sequence.get(sequence,[])):
            # v6 URLs verified against the public API on 2026-09-21. A v4
            # fallback may retain the benchmark-era sequence. Every file is
            # verified independently; version alone never establishes identity.
            urls=[f'https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v{v}.pdb' for v in (6,4)]
            for url in urls:
                try:
                    content=dl.get(url)
                    if pdb_sequence(content.decode())!=sequence:
                        attempts.append(dict(url=url,error='coordinate sequence mismatch or incomplete structure'));continue
                    temp=pdb.with_suffix('.partial');temp.write_bytes(content);temp.replace(pdb)
                    meta=dict(protein_id=pid,uniprot_id=uid,path=str(pdb.relative_to(ROOT)),url=url,
                        bytes=len(content),sha256=hashlib.sha256(content).hexdigest(),
                        sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest(),
                        exact_sequence_match=True,residues=len(sequence),retrieved_unix=time.time())
                    receipt.write_text(json.dumps(meta,indent=2)+'\n');return meta
                except Exception as error:
                    attempts.append(dict(url=url,error=str(error)))
            # Latest API URLs can differ; do not assume a missing v6 entry has
            # no publicly released structure.
            try:
                metadata=json.loads(dl.get('https://alphafold.ebi.ac.uk/api/prediction/'+uid))
                for model in metadata:
                    if model.get('uniprotSequence')!=sequence or not model.get('pdbUrl'):continue
                    url=model['pdbUrl']
                    if url in urls:continue
                    content=dl.get(url)
                    if pdb_sequence(content.decode())!=sequence:continue
                    pdb.write_bytes(content)
                    meta=dict(protein_id=pid,uniprot_id=uid,path=str(pdb.relative_to(ROOT)),url=url,
                        bytes=len(content),sha256=hashlib.sha256(content).hexdigest(),
                        sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest(),
                        exact_sequence_match=True,residues=len(sequence),retrieved_unix=time.time())
                    receipt.write_text(json.dumps(meta,indent=2)+'\n');return meta
            except Exception as error:attempts.append(dict(uid=uid,error=str(error)))
        return dict(protein_id=pid,exact_sequence_match=False,attempts=attempts)
    counts=dict(total=len(records),checked=0,exact_structures=0,unresolved=0,priority_missing_pockets_resolved=0)
    started=time.time();manifest=root/('manifest_pilot.jsonl' if a.limit else 'manifest.jsonl')
    with manifest.open('w') as log,concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        for meta in pool.map(download,records):
            log.write(json.dumps(meta)+'\n');log.flush();counts['checked']+=1
            if meta['exact_sequence_match']:
                counts['exact_structures']+=1
                if meta['protein_id'] in missing:counts['priority_missing_pockets_resolved']+=1
            else:counts['unresolved']+=1
            if counts['checked']%100==0 or counts['checked']==len(records):
                status=dict(**counts,seconds=time.time()-started,stage='complete' if counts['checked']==len(records) else 'downloading',
                    purpose='Exact-sequence structural inputs; pocket extraction and native graph preprocessing remain separate steps')
                target=root/('pilot_status.json' if a.limit else 'status.json')
                temp=target.with_suffix('.tmp');temp.write_text(json.dumps(status,indent=2)+'\n');temp.replace(target)
                print(json.dumps(status),flush=True)


if __name__=='__main__':main()
