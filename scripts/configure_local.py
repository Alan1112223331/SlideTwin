"""Create local config by reference; never copy or print an API key."""
import argparse
import json
from pathlib import Path
import re

parser = argparse.ArgumentParser()
parser.add_argument("credential_file", type=Path)
parser.add_argument("--model", default="Qwen/Qwen3-VL-30B-A3B-Instruct")
parser.add_argument("--base-url", help="Explicit provider endpoint when the key file contains only a key")
parser.add_argument("--out", type=Path, default=Path("config.local.toml"))
args = parser.parse_args()
credential = args.credential_file.resolve()
text = credential.read_text(encoding="utf-8-sig")
urls = re.findall(r"https?://[^\s]+", text)
if not urls and not args.base_url:
    raise SystemExit("Credential file must contain an endpoint URL, or configure base_url manually")
base = (args.base_url or urls[0]).rstrip("/")
if not base.endswith("/v1"):
    base += "/v1"
template = Path(__file__).resolve().parents[1]/"config.example.toml"
config = template.read_text(encoding="utf-8")
for key,value in [('base_url',base),('model',args.model),('api_key_file',str(credential))]:
    config=re.sub(r'^'+key+r' = .*$',lambda _:key+' = '+json.dumps(value,ensure_ascii=False),config,flags=re.M)
config=re.sub(r'^fallback_models = .*$', 'fallback_models = [] # Configure compatible fallback limits explicitly',config,flags=re.M)
args.out.write_text(config, encoding="utf-8")
print(f"Saved {args.out}; credentials remain in the original private file")
