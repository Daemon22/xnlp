#!/usr/bin/env python
"""Search archive.org simple API for Masikhanyise and isiXhosa books."""
import sys, io, re, json, ssl
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', write_through=True)
from urllib.request import urlopen, Request

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# Simple search
for term in ['masikhanyise isixhosa', 'masikhanyise xhosa', 'maskew miller', 'isiXhosa textbook']:
    url = f"https://archive.org/advanced_search.php?q={term.replace(' ', '+')}&output=json&rows=10&page=1"
    try:
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        resp = urlopen(req, timeout=20, context=ctx)
        data = json.loads(resp.read().decode('utf-8'))
        docs = data.get('response', {}).get('docs', [])
        print(f'\nSearch "{term}": {len(docs)} results')
        for doc in docs[:5]:
            ident = doc.get('identifier', '?')
            title = doc.get('title', '?')
            creator = doc.get('creator', '?')
            year = doc.get('year', '?')
            print(f'  {ident}: {title[:80]} by {creator[:40]} ({year})')
    except Exception as e:
        print(f'\nSearch "{term}": {e}')

# Also try the main archive.org search page
print("\n=== Archive.org main search ===")
url = "https://archive.org/search.php?query=masikhanyise"
try:
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    resp = urlopen(req, timeout=20, context=ctx)
    html = resp.read().decode('utf-8', errors='replace')
    
    # Find result items
    items = re.findall(r'href="/details/([^"]+)"', html)
    print(f'Found {len(items)} potential items')
    for item in items[:10]:
        print(f'  /details/{item}')
except Exception as e:
    print(f'Error: {e}')

# Try searching for any isiXhosa books on archive.org
print("\n=== Archive.org: isiXhosa language books ===")
url = "https://archive.org/search.php?query=language%3Axho&sort=-publicdate"
try:
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    resp = urlopen(req, timeout=20, context=ctx)
    html = resp.read().decode('utf-8', errors='replace')
    
    items = re.findall(r'href="/details/([^"]+)"', html)
    print(f'Found {len(items)} isiXhosa items')
    for item in items[:15]:
        print(f'  /details/{item}')
except Exception as e:
    print(f'Error: {e}')
