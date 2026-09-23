"""Parallel page translation with full context and independent repair checkpoints."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
import re

from .extract import restore
from .models import Page, digest, write_json, read_cache
from .protocol import ProtocolError, decode_response, parse_content_response as parse_response, response_format, restore_plain_tokens, aligned_phrase
from .translate import Translator, SYSTEM
from .client import ProviderError


ASYNC_VERSION = "2"


class PageGroupError(ProtocolError):
    """Report only the individual pages still failing after bounded recovery."""
    def __init__(self, translations, failures):
        self.translations, self.failures = translations, failures
        super().__init__(f"{len(failures)} pages remain incomplete after automatic recovery")


class AsyncTranslator(Translator):
    def page_content_key(self,page):
        # Optional terminology preparation may improve on resume. That alone
        # must not invalidate completed, reviewed pages from this same input.
        return digest(ASYNC_VERSION+self.config.fingerprint()+self.document.source_sha256+self.context+json.dumps(asdict(page),sort_keys=True))

    def page_cache(self,page,key):
        path=self.cache/f'page-{page.number:04d}.json'
        cached=read_cache(path)
        if cached.get('key')==key and isinstance(cached.get('translations'),dict):return cached
        if (cached.get('content_key')==self.page_content_key(page) and isinstance(cached.get('translations'),dict)
                and cached.get('reviewed')==self.config.translation.review):
            return {**cached,'key':key}
        ledger=getattr(self,'resume_ledger',{})
        if (ledger.get('source_sha256')==self.document.source_sha256 and ledger.get('config_fingerprint')==self.config.fingerprint()
                and ledger.get('page_keys',{}).get(str(page.number))==key and page.number in ledger.get('completed_pages',[])):
            try:values=parse_response(json.dumps({k:ledger.get('translations',{}).get(k) for k in self.sources(page)}),self.sources(page),'json')
            except ProtocolError:return {}
            cached={'key':key,'page':page.number,'translations':values,'reviewed':self.config.translation.review,'recovered_from':'translation-ledger.json'}
            write_json(path,cached)
            return cached
        return {}

    def grouped_page(self,numbers):
        pages=[self.document.pages[n-1] for n in numbers]
        return Page(pages[0].number,pages[0].width,pages[0].height,[r for p in pages for r in p.regions],
                    '\n\n'.join(f'PAGE {p.number}\n{p.context}' for p in pages),
                    ocr_used=any(self.page_uses_ocr(p) for p in pages))

    def estimate_batch(self,numbers):
        group=self.grouped_page(numbers);sources=self.sources(group)
        mode='tagged' if self.config.provider.protocol=='auto' else self.config.provider.protocol
        messages=self.messages(group,sources,mode,include_images=False)
        input_tokens=self.counter.request(messages,response_format(sources,mode))
        image_pages=set()
        if self.config.provider.vision and self.config.translation.image_policy=='ocr_only':
            for number in numbers:
                for n in range(max(1,number-self.config.translation.image_neighbors),min(len(self.document.pages),number+self.config.translation.image_neighbors)+1):
                    if self.page_uses_ocr(self.document.pages[n-1]):image_pages.add(n)
            for n in image_pages:
                p=self.document.pages[n-1];dpi=self.config.translation.image_dpi
                input_tokens+=self.counter.image(p.width*dpi/72,p.height*dpi/72)+32
            if image_pages:input_tokens+=128
        output=self.counter.output(sources,self.config.translation.output_expansion_ratio)
        review_input=input_tokens+output+4096+128 if self.config.translation.review else input_tokens
        capacity=self.counter.admission_capacity
        fits=review_input+output<=capacity and output<=min(self.config.provider.max_output_tokens,self.config.provider.model_max_output_tokens)
        return {'pages':numbers,'input_tokens_estimated':input_tokens,'review_input_tokens_estimated':review_input,
                'output_tokens_reserved':output,'context_total_estimated':review_input+output,
                'context_capacity':self.counter.capacity,'admission_capacity':capacity,'ocr_image_pages':sorted(image_pages),'fits':fits}

    def plan_batches(self,selected):
        chunks=[];estimates=[];start=0
        while start<len(selected):
            limit=len(selected)-start
            if self.config.translation.batch_mode=='fixed':limit=min(limit,self.config.translation.pages_per_request)
            low,high,best=1,limit,None
            while low<=high:
                mid=(low+high)//2;estimate=self.estimate_batch(selected[start:start+mid])
                if estimate['fits']:best=estimate;low=mid+1
                else:high=mid-1
            if best is None:
                estimate=self.estimate_batch(selected[start:start+1])
                # Keep an oversized page independent. It may already have a
                # valid checkpoint; otherwise its own call reports the limit.
                best=estimate
                self.log(f"Page {selected[start]} exceeds request capacity; isolated from other groups")
            chunks.append(best['pages']);estimates.append(best);start+=len(best['pages'])
        write_json(self.work/'batch-plan.json',{'mode':self.config.translation.batch_mode,'model':self.config.provider.model,
                   'context_utilization':self.config.provider.context_utilization,'batches':estimates})
        self.log(f'Planned {len(chunks)} request groups for {len(selected)} pages using context/output budgets (no fixed 4-page limit)')
        return chunks

    def recurring_key(self,region):
        page=self.document.pages[region.page-1]
        if not (region.bbox[3]<page.height*.08 or region.bbox[1]>page.height*.93):return None
        # Only identical recurring page furniture, including every immutable
        # value and style phrase; body passages always retain their own context.
        return digest(json.dumps({'source':region.source,'protected':region.protected,'styles':region.inline_styles},sort_keys=True))

    def remember_recurring(self,page,values,path):
        for region in page.regions:
            key=self.recurring_key(region)
            if key is None or self.recurring_counts[key]<3:continue
            sources={k:v for k,v in self.sources(page).items() if k==region.id or k.startswith(region.id+'_s')}
            if not set(sources)<=set(values):continue
            group={k:values[k] for k in sources}
            self.recurring[key]={'values':{k[len(region.id):]:v for k,v in group.items()},'cache':str(path),'cache_sha256':digest(path.read_bytes())}

    def load_recurring(self):
        from collections import Counter
        self.recurring_counts=Counter(self.recurring_key(r) for p in self.document.pages for r in p.regions)
        self.recurring={};self.recurring_used={}
        for page in self.document.pages:
            path=self.cache/f'page-{page.number:04d}.json'
            if not path.exists():continue
            obj=read_cache(path)
            key=digest(ASYNC_VERSION+self.config.fingerprint()+self.document.source_sha256+self.context+self.glossary+json.dumps(asdict(page),sort_keys=True))
            if obj.get('key')!=key or not obj.get('reviewed'):continue
            try:
                values=parse_response(json.dumps(obj.get('translations',{})),self.sources(page),'json')
                self.remember_recurring(page,values,path)
            except ProtocolError:continue

    def recurring_values(self,page,sources):
        values={}
        for region in page.regions:
            key=self.recurring_key(region)
            if key is None:continue
            record=getattr(self,'recurring',{}).get(key)
            if not record:continue
            group={region.id+suffix:v for suffix,v in record['values'].items()}
            if not set(group)<=set(sources):continue
            values.update(group);self.recurring_used[region.id]={**record,'source_key':key}
        if values:write_json(self.work/'recurring-translation-provenance.json',self.recurring_used)
        return values

    def event(self, page, kind, **details):
        entry = {"page": page.number, "kind": kind, **details}
        self.events.append(entry)
        write_json(self.work/"translation-events.json", self.events)

    async def call(self, page, sources, mode, draft=None, feedback=""):
        messages=self.messages(page,sources,mode,draft,feedback)
        expected=self.counter.output(sources,self.config.translation.output_expansion_ratio)
        input_tokens=self.counter.request(messages,response_format(sources,mode))
        try:limit=self.counter.request_limit(input_tokens,expected)
        except ValueError as exc:raise ProtocolError(str(exc)) from exc
        if self.config.provider.tokens_per_minute and input_tokens+limit>int(self.config.provider.tokens_per_minute*self.config.provider.token_rate_utilization):
            raise ProtocolError('Request exceeds configured TPM admission budget')
        text=await self.client.complete(messages, response_format(sources, mode),max_output_tokens=limit,
                                          expected_output_tokens=limit,
                                          label={"document": self.source.stem, "page": page.number, "mode": mode, "has_draft": draft is not None})
        self.retain_candidate(page,sources,mode,text,draft)
        return text

    async def review_async(self,page,sources,draft):
        """Split only for measured token capacity, never because of meaning."""
        mode='tagged' if self.config.provider.protocol=='auto' else self.config.provider.protocol
        input_tokens=self.counter.request(self.messages(page,sources,mode,draft),response_format(sources,mode))
        output_tokens=self.counter.output(sources,self.config.translation.output_expansion_ratio)
        capacity=self.counter.admission_capacity
        numbers=sorted({r.page for r in page.regions})
        if input_tokens+output_tokens>capacity and len(numbers)>1:
            self.event(page,'review_capacity_split',pages=numbers,input_tokens=input_tokens,output_tokens=output_tokens)
            async def half(ns):
                group=self.grouped_page(ns);ids=self.sources(group)
                return await self.review_async(group,ids,{k:draft[k] for k in ids})
            values=await asyncio.gather(half(numbers[:len(numbers)//2]),half(numbers[len(numbers)//2:]),return_exceptions=True)
            accepted=self.batch_checkpoint(page,sources,self.config.provider.protocol,draft)
            for result in values:
                if isinstance(result,dict):accepted.update(result)
            # Completed halves survive a sibling failure, including recursive
            # splits. Recovery must not review them a second time.
            for ns in (numbers[:len(numbers)//2],numbers[len(numbers)//2:]):
                group=self.grouped_page(ns);ids=self.sources(group)
                accepted.update(self.batch_checkpoint(group,ids,self.config.provider.protocol,{k:draft[k] for k in ids}))
            key,path=self.batch_identity(page,sources,self.config.provider.protocol,draft)
            write_json(path,{'key':key,'page':page.number,'complete':len(accepted)==len(sources),'translations':accepted})
            self.accepted_candidates={**getattr(self,'accepted_candidates',{}),**accepted}
            errors=[r for r in values if isinstance(r,BaseException)]
            if errors:raise errors[0]
            return accepted
        return await self.batch_async(page,sources,self.config.provider.protocol,draft)

    async def glossary_async(self):
        if not self.config.translation.glossary:
            return
        key = digest(ASYNC_VERSION+self.config.fingerprint()+self.document.source_sha256+self.context)
        path = self.cache/"glossary.json"
        if path.exists():
            saved=read_cache(path)
            if saved.get("key")==key and saved.get('complete',True) and isinstance(saved.get('text'),str):
                self.glossary=saved["text"]
                return
        prefix=(f"Translate terminology into {self.config.translation.target_language}. Read the supplied course context. "
                "Return only a concise guide of at most 60 recurring technical terms (source -> target), "
                "plus essential distinctions. At most 1800 Chinese characters. No page-by-page summary, no layout instructions.\nCOURSE CONTEXT:\n")
        self.log("Preparing document-wide terminology")
        capacity=self.counter.admission_capacity
        output_limit=min(2500,self.config.provider.max_output_tokens,self.config.provider.model_max_output_tokens)
        overhead=self.counter.messages([{'role':'system','content':SYSTEM},{'role':'user','content':prefix}])+output_limit
        if overhead>=capacity:raise ProtocolError('Configured token budget is too small for terminology preparation')
        # Normally one full-document request. Oversized documents keep all text
        # in sequential context chunks, dispatched independently.
        chunks=[];remaining=self.context
        while remaining:
            low,high,best=1,len(remaining),0
            while low<=high:
                mid=(low+high)//2
                if self.counter.text(remaining[:mid])+overhead<=capacity:best=mid;low=mid+1
                else:high=mid-1
            if not best:raise ProtocolError('No terminology context fits the configured budget')
            if best<len(remaining):
                boundary=remaining.rfind('\n',0,best)
                if boundary>best//2:best=boundary+1
            chunks.append(remaining[:best]);remaining=remaining[best:]
        async def guide(index,text):
            part_key=digest(key+text)
            part_path=self.cache/'glossary-parts'/f'{part_key}.json'
            saved=read_cache(part_path)
            if saved.get('key')==part_key and isinstance(saved.get('text'),str) and saved['text'].strip():return saved['text']
            value=await self.client.complete([{"role":"system","content":SYSTEM},{"role":"user","content":prefix+text}],
                max_output_tokens=output_limit,label={"document":self.source.stem,"phase":"glossary","context_part":index+1,"context_parts":len(chunks)})
            write_json(part_path,{'key':part_key,'text':value})
            return value
        results=await asyncio.gather(*(guide(i,text) for i,text in enumerate(chunks)),return_exceptions=True)
        failures=[{'part':i+1,'error':str(r)} for i,r in enumerate(results) if isinstance(r,Exception)]
        for result in results:
            if isinstance(result,BaseException) and not isinstance(result,(ProviderError,ProtocolError)):raise result
            if isinstance(result,ProviderError) and not result.allows_page_recovery:raise result
        self.glossary='\n\n'.join(r for r in results if isinstance(r,str))
        write_json(path,{'key':key,'text':self.glossary,'complete':not failures,'failures':failures})
        if failures:
            self.preparation_warnings=getattr(self,'preparation_warnings',[])+[{'stage':'glossary','failures':failures}]
            self.log('Terminology preparation partly failed; retaining completed guides and continuing translation with course context')

    def groups(self, sources):
        groups={}
        for key in sources:
            parent=re.sub(r"_s\d+$","",key)
            groups.setdefault(parent, {})[key]=sources[key]
        return list(groups.values())

    def accept_groups(self, page, sources, raw, draft):
        accepted, reasons = {}, []
        if not isinstance(raw,dict):raw={}
        protected={r.id:r.protected for r in page.regions}
        for key,source in sources.items():
            group={key:source}
            if not set(group)<=set(raw):
                reasons.append("Missing targets: "+", ".join(sorted(set(group)-set(raw))))
                continue
            try:
                values=parse_response(json.dumps({k:raw[k] for k in group}),group,"json",protected)
                accepted.update(values)
            except ProtocolError as exc:
                reasons.append(str(exc))
        return accepted, reasons

    async def plain(self, page, key, source, draft):
        region=next((r for r in page.regions if r.id==key),None)
        cache_key=digest(ASYNC_VERSION+self.config.fingerprint()+self.document.source_sha256+self.context+self.glossary+
                         json.dumps({'page':asdict(page),'target':key,'source':source,'draft':draft},sort_keys=True))
        cache_path=self.cache/'plain-targets'/f'{cache_key}.json'
        if cache_path.exists():
            saved=read_cache(cache_path)
            if saved.get('key')==cache_key:
                try:
                    value=parse_response(json.dumps({key:saved.get('translation')}),{key:source},'json')[key]
                except ProtocolError:
                    pass
                else:
                    return value
        # Author glyphs are source-owned; translating surrounding fields must not
        # require the model to output an exact spelling of an immutable identity.
        authors=[token for token,asset in (region.protected_assets.items() if region else [])
                 if token in source and re.search(r"\b(?:Dr|Prof)\.",asset["text"])]
        if authors:
            parts=re.split("("+"|".join(re.escape(t) for t in authors)+")",source)
            output=[]
            for part_index,part in enumerate(parts):
                if part in authors or not part.strip(' |'):
                    output.append(part)
                else:
                    lead=re.match(r"^[\s|]*",part)[0];trail=re.search(r"[\s|]*$",part)[0]
                    core=part[len(lead):len(part)-len(trail) if trail else None]
                    output.append(lead+(await self.plain(page,f'{key}__part{part_index}',core,draft)).strip(' |')+trail)
            return ''.join(output)
        sources={key:source}
        literal=restore(region,source) if region else source
        feedback=""
        for attempt in range(3):
            text=await self.call(page,{key:literal},"plain",draft,feedback)
            try:
                value=parse_response(text,{key:literal},"plain")[key]
                if region:
                    try:value=restore_plain_tokens(value,{t:v for t,v in region.protected.items() if t in source},source)
                    except ProtocolError:pass
                write_json(cache_path,{'key':cache_key,'translation':value})
                return value
            except ProtocolError as exc:
                feedback=str(exc)
                self.event(page,"plain_repair",target=key,reason=feedback,attempt=attempt+1)
        raise ProtocolError(feedback)

    def batch_identity(self,page,sources,mode,draft=None):
        key=digest(ASYNC_VERSION+self.config.fingerprint()+self.document.source_sha256+self.context+self.glossary+
                   json.dumps({"page":asdict(page),"sources":sources,"mode":mode,"draft":draft},sort_keys=True))
        return key,self.cache/"async-batches"/f"{key}.json"

    def batch_checkpoint(self,page,sources,mode,draft=None):
        key,path=self.batch_identity(page,sources,mode,draft)
        if not path.exists():return {}
        saved=read_cache(path)
        if saved.get('key')!=key:return {}
        return self.accept_groups(page,sources,saved.get('translations',{}),draft)[0]

    async def batch_async(self,page,sources,mode,draft=None):
        key,path=self.batch_identity(page,sources,mode,draft)
        accepted=self.recurring_values(page,sources)
        if path.exists():
            saved=read_cache(path)
            if saved.get('key')==key:
                cached,_=self.accept_groups(page,sources,saved.get('translations',{}),draft)
                accepted.update(cached)
        pending={k:v for k,v in sources.items() if k not in accepted}
        def checkpoint():
            write_json(path,{"key":key,"page":page.number,"complete":len(accepted)==len(sources),"translations":accepted})
            self.accepted_candidates={**getattr(self,'accepted_candidates',{}),**accepted}
        checkpoint()
        feedback=""
        if mode!='plain':
            for attempt in range(3):
                if not pending:
                    break
                context_draft={**(draft or {}),**accepted} or None
                text=await self.call(page,pending,"tagged" if mode=='auto' else mode,context_draft,feedback)
                try:
                    raw=parse_response(text,pending,"tagged" if mode=='auto' else mode,allow_missing=True)
                    good,reasons=self.accept_groups(page,pending,raw,context_draft)
                    accepted.update(good)
                    pending={k:v for k,v in sources.items() if k not in accepted}
                    feedback='; '.join(reasons)
                    ocr=[r.id for r in page.regions if not r.native and r.id in pending]
                    if ocr:
                        feedback += (' The following IDs came from OCR and may contain recognition errors: '+', '.join(ocr)+
                            '. Read the corresponding original page images to identify the actual visible words. Translate the visible label faithfully; do not retain or literally translate a garbled OCR syllable. Return the same requested IDs.')
                    checkpoint()
                except ProtocolError as exc:
                    feedback=str(exc)
                if pending:
                    self.event(page,"targeted_repair",attempt=attempt+1,remaining=list(pending),reason=feedback)
                    write_json(self.cache/f"rejected-async-{page.number:04d}-{len(self.events):05d}.json",{"reason":feedback,"response":text})
            if pending and mode not in {'auto','plain'}:
                raise ProtocolError(feedback)
        if pending:
            self.event(page,"plain_fallback",remaining=list(pending))
            # Independent sentences are concurrent; a styled child waits for its
            # own parent sentence. Every call still sees the entire course.
            async def group_work(group):
                local={**(draft or {}),**accepted}
                values={}
                for target,source in group.items():
                    value=await self.plain(page,target,source,{**local,**values} or None)
                    values[target]=value
                    accepted[target]=value
                    checkpoint()
            results=await asyncio.gather(*(group_work(g) for g in self.groups(pending)),return_exceptions=True)
            errors=[r for r in results if isinstance(r,BaseException)]
            if errors:
                raise errors[0]
        hints=self.language_hints(page,sources,accepted)
        if hints:self.event(page,'advisory_language_hints',phase='review' if draft is not None else 'translation',hints=hints)
        return accepted

    async def page_async(self,number):
        page=self.document.pages[number-1]
        sources=self.sources(page)
        key=digest(ASYNC_VERSION+self.config.fingerprint()+self.document.source_sha256+self.context+self.glossary+json.dumps(asdict(page),sort_keys=True))
        path=self.cache/f"page-{number:04d}.json"
        if path.exists() or getattr(self,'resume_ledger',{}):
            saved=self.page_cache(page,key)
            if saved.get('key')==key:
                try:
                    values=parse_response(json.dumps(saved.get('translations',{})),sources,'json')
                except ProtocolError:
                    pass
                else:
                    self.log(f"Page {number}: verified cache")
                    return values
        self.log(f"Page {number}: translating {len(sources)} targets")
        draft=await self.batch_async(page,sources,self.config.provider.protocol) if sources else {}
        final=await self.review_async(page,sources,draft) if sources and self.config.translation.review else draft
        write_json(path,{"key":key,"page":number,"model":self.config.provider.model,"protocol":self.config.provider.protocol,
                         'content_key':self.page_content_key(page),
                         "worker_models":self.config.provider.worker_models or [self.config.provider.model],
                         "reviewed":self.config.translation.review,"draft":draft,"translations":final,"engine":"async-"+ASYNC_VERSION})
        self.log(f"Page {number}: complete and reviewed")
        return final

    async def run_async(self,selected):
        self.resume_ledger=read_cache(self.work/'translation-ledger.json')
        self.preparation_warnings=[]
        try:
            await self.glossary_async()
        except (ProtocolError,ProviderError) as exc:
            if isinstance(exc,ProviderError) and not exc.allows_page_recovery:raise
            self.preparation_warnings.append({'stage':'glossary','error':str(exc)})
            self.log(f'Terminology preparation unavailable; continuing with course context: {exc}')
        self.load_recurring()
        chunks=self.plan_batches(selected)
        async def work(numbers):
            try:
                if len(numbers)==1:
                    result=await self.page_async(numbers[0])
                else:result=await self.pages_async(numbers)
            except Exception as exc:result=exc
            return numbers,result
        translations,failures={},[]
        completed=[]
        page_keys={str(n):digest(ASYNC_VERSION+self.config.fingerprint()+self.document.source_sha256+self.context+self.glossary+json.dumps(asdict(self.document.pages[n-1]),sort_keys=True)) for n in selected}
        def save():
            write_json(self.work/'translation-failures.json',failures)
            write_json(self.work/'translation-ledger.json',{'source_sha256':self.document.source_sha256,'config_fingerprint':self.config.fingerprint(),
                'selected_pages':selected,'completed_pages':sorted(completed),'translations':translations,'accepted_candidates':getattr(self,'accepted_candidates',{}),
                'usage':self.client.usage,'failures':failures,'preparation_warnings':self.preparation_warnings,'page_keys':page_keys})
        # Every group can run independently. The transport admits new requests
        # as each slot/quota becomes available; there is no batch-wide barrier.
        for future in asyncio.as_completed([asyncio.create_task(work(ns)) for ns in chunks]):
            numbers,result=await future
            if isinstance(result,PageGroupError):
                translations.update(result.translations)
                failures.extend(result.failures)
                failed={f['page'] for f in result.failures};completed.extend(n for n in numbers if n not in failed)
            elif isinstance(result,BaseException):
                failures.extend({"page":number,"type":type(result).__name__,"error":str(result)} for number in numbers)
            else:
                translations.update(result)
                completed.extend(numbers)
            save()
        if failures:
            raise ProtocolError(f"{len(failures)} pages remain incomplete; {len(selected)-len(failures)} pages complete and retained; see translation-failures.json for exact pages and causes")
        return translations

    async def pages_async(self,numbers):
        pages=[self.document.pages[n-1] for n in numbers]
        keys={p.number:digest(ASYNC_VERSION+self.config.fingerprint()+self.document.source_sha256+self.context+self.glossary+json.dumps(asdict(p),sort_keys=True)) for p in pages}
        cached={};pending=[]
        for p in pages:
            path=self.cache/f'page-{p.number:04d}.json'
            if path.exists() or getattr(self,'resume_ledger',{}):
                obj=self.page_cache(p,keys[p.number])
                if obj.get('key')==keys[p.number]:
                    try:
                        v=parse_response(json.dumps(obj.get('translations',{})),self.sources(p),'json')
                    except ProtocolError:pass
                    else:
                        cached.update(v);continue
            pending.append(p)
        if not pending:return cached
        if len(pending)==1:
            try:return {**cached,**(await self.page_async(pending[0].number))}
            except Exception as exc:
                raise PageGroupError(cached,[{'page':pending[0].number,'type':type(exc).__name__,'error':str(exc)}]) from exc
        group=self.grouped_page([p.number for p in pending])
        sources=self.sources(group)
        self.log(f'Pages {[p.number for p in pending]}: translating {len(sources)} targets together')
        draft=None
        try:
            draft=await self.batch_async(group,sources,self.config.provider.protocol) if sources else {}
            final=await self.review_async(group,sources,draft) if sources and self.config.translation.review else draft
        except (ProtocolError,ProviderError) as exc:
            if isinstance(exc,ProviderError) and not exc.allows_page_recovery:
                raise PageGroupError(cached,[{'page':p.number,'type':type(exc).__name__,'error':str(exc)} for p in pending]) from exc
            return await self.recover_pages(group,pending,sources,draft,cached,exc)
        for p in pending:
            ids=self.sources(p)
            write_json(self.cache/f'page-{p.number:04d}.json',{'key':keys[p.number],'page':p.number,'model':self.config.provider.model,
                'content_key':self.page_content_key(p),
                'worker_models':self.config.provider.worker_models or [self.config.provider.model],
                'protocol':self.config.provider.protocol,'reviewed':self.config.translation.review,'draft':{k:draft[k] for k in ids},
                'translations':{k:final[k] for k in ids},'engine':'async-'+ASYNC_VERSION,'request_pages':[x.number for x in pending]})
            self.log(f'Page {p.number}: complete and reviewed')
            self.remember_recurring(p,{k:final[k] for k in ids},self.cache/f'page-{p.number:04d}.json')
        return {**cached,**final}

    async def recover_pages(self,group,pages,sources,draft,cached,cause):
        """One bounded page-level pass, retaining validated grouped checkpoints."""
        mode=self.config.provider.protocol
        initial=draft if draft is not None else self.batch_checkpoint(group,sources,mode)
        reviewed=self.batch_checkpoint(group,sources,mode,draft) if draft is not None and self.config.translation.review else {}
        self.log(f'Pages {[p.number for p in pages]}: grouped request incomplete ({cause}); automatically recovering individual pages with context')
        self.event(group,'automatic_page_recovery',pages=[p.number for p in pages],reason=str(cause))
        for page in pages:
            ids=self.sources(page)
            page_draft={k:initial[k] for k in ids if k in initial}
            for values,parent_draft in [(page_draft,None),({k:reviewed[k] for k in ids if k in reviewed},page_draft)]:
                if not values or (parent_draft is not None and len(page_draft)!=len(ids)):continue
                good,_=self.accept_groups(page,ids,values,parent_draft)
                if not good:continue
                key,path=self.batch_identity(page,ids,mode,parent_draft)
                good={**self.batch_checkpoint(page,ids,mode,parent_draft),**good}
                _,origin=self.batch_identity(group,sources,mode,draft if parent_draft is not None else None)
                write_json(path,{'key':key,'page':page.number,'complete':len(good)==len(ids),'translations':good,
                                 'recovered_from_batch':str(origin),'origin_sha256':digest(origin.read_bytes())})
        results=await asyncio.gather(*(self.page_async(p.number) for p in pages),return_exceptions=True)
        complete=dict(cached);failures=[]
        for page,result in zip(pages,results):
            if isinstance(result,BaseException):
                failures.append({'page':page.number,'type':type(result).__name__,'error':str(result)})
            else:complete.update(result)
        if failures:raise PageGroupError(complete,failures)
        return complete
