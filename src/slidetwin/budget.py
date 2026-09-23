"""Token-aware request sizing; output reservation is part of the context budget."""
from __future__ import annotations

import math
import json
from pathlib import Path
import re


class TokenCounter:
    def __init__(self,provider):
        self.provider=provider;self.tokenizer=None
        if provider.tokenizer_file:
            from tokenizers import Tokenizer
            self.tokenizer=Tokenizer.from_file(str(Path(provider.tokenizer_file)))

    def text(self,value):
        if self.tokenizer is not None:return len(self.tokenizer.encode(value,add_special_tokens=False).ids)
        # Conservative fallback for arbitrary OpenAI-compatible models.
        cjk=len(re.findall(r'[\u3400-\u9fff]',value))
        return math.ceil(cjk*1.3+(len(value)-cjk)/2.5)

    def messages(self,messages):
        total=3
        for message in messages:
            total+=8
            content=message.get('content','')
            if isinstance(content,str):total+=self.text(content)
            else:
                for part in content:
                    if part.get('type')=='text':total+=self.text(part.get('text',''))+2
                    elif part.get('type')=='image_url':
                        import base64,io
                        from PIL import Image
                        try:
                            with Image.open(io.BytesIO(base64.b64decode(part['image_url']['url'].split(',',1)[1]))) as im:
                                total+=self.image(im.width,im.height)
                        except Exception:total+=4096
        return total

    def request(self,messages,response_format=None):
        # A strict JSON schema is also provider-visible prompt input.
        return self.messages(messages)+(self.text(json.dumps(response_format,ensure_ascii=False))+8 if response_format else 0)

    @staticmethod
    def image(width,height):
        # Conservative 28 px grid, including framing. Server resizing can differ;
        # actual prompt usage is separately recorded in the request trace.
        return math.ceil(width/28)*math.ceil(height/28)+85

    @property
    def capacity(self):return math.floor(self.provider.context_window_tokens*self.provider.context_utilization)

    @property
    def admission_capacity(self):
        return min(self.capacity,math.floor(self.provider.tokens_per_minute*self.provider.token_rate_utilization)) if self.provider.tokens_per_minute else self.capacity

    def output(self,sources,ratio=1.5):
        return max(128,math.ceil(sum(self.text(v)*ratio+self.text(k)+14 for k,v in sources.items())+64))

    def request_limit(self,input_tokens,estimated_output):
        available=min(self.provider.max_output_tokens,self.provider.model_max_output_tokens,self.capacity-input_tokens)
        if estimated_output>available:
            raise ValueError(f'Request needs {input_tokens} input + {estimated_output} output tokens; configured context/output budget cannot fit it')
        return min(available,max(128,estimated_output))
