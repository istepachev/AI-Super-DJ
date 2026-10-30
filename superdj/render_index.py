"""Small content-addressed render decisions; never substitute metadata for PCM QA."""
from copy import deepcopy
from pathlib import Path
import hashlib,json,threading,time
from contextlib import contextmanager
from collections import OrderedDict
from contextvars import ContextVar

_correction_budget=ContextVar('superdj_indexed_backend_corrections',default=None)


def indexed_compensate(function,*args,**options):
    budget=_correction_budget.get()
    if budget is not None and budget['used']>=budget['maximum']:
        raise ValueError('Candidate backend-correction budget exhausted; retain the best audited result or choose another window')
    value=function(*args,**options)
    if budget is not None:budget['used']+=1
    return value


class RenderSearchBudgetExceeded(RuntimeError):
    def __init__(self,info):
        self.info=info
        super().__init__('Render search budget exhausted for '+', '.join(info['source_ids'])+
                         '; choose another compatible window/neighbour or optional source; audio checks remain required')


def clip_dependencies(value):
    """The rebasing tag is bookkeeping; all actual edited features stay keyed.

    Keep the tag in the renderer record so old saved voice timelines remain
    idempotent. Its hash contains temporary paths, unlike content-checked file
    references and the actual rebased profile that still participate below.
    """
    row=value.get('record',{})
    if 'render_evidence_timeline_signature'not in row:return value
    result=dict(value);result['record']=dict(row)
    result['record'].pop('render_evidence_timeline_signature')
    return result


def _packed(value):
    return json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':'))


class RenderIndex:
    def __init__(self,session,scope,*,track_count=1,max_group_candidates=6,
                 recovery_seconds=180.,max_extra_candidates=None,max_backend_corrections=3):
        if not isinstance(scope,str)or not scope:raise ValueError('Render index needs an explicit programme scope')
        if isinstance(max_group_candidates,bool)or not isinstance(max_group_candidates,int)or not 1<=max_group_candidates<=16:
            raise ValueError('Render candidate limit must be 1–16')
        if not 1<=recovery_seconds<=3600:raise ValueError('Invalid recovery time budget')
        self.session=session;self.scope=hashlib.sha256(scope.encode()).hexdigest()
        self.limit=max_group_candidates;self.seconds=float(recovery_seconds)
        if isinstance(max_backend_corrections,bool)or not isinstance(max_backend_corrections,int)or not 0<=max_backend_corrections<=9:
            raise ValueError('Backend correction budget must be 0–9')
        self.backend_limit=max_backend_corrections
        self.extra_limit=max(16,2*track_count)if max_extra_candidates is None else max_extra_candidates
        if isinstance(self.extra_limit,bool)or not isinstance(self.extra_limit,int)or self.extra_limit<0:raise ValueError('Invalid programme retry budget')
        self.memory=OrderedDict();self.memory_sizes={};self.memory_bytes=0;self.memory_limit=16*1024**2
        self.lock=threading.RLock();self.key_locks={}
        self.token=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:20]
        policy=_packed({'candidates':self.limit,'seconds':self.seconds,'extra':self.extra_limit,'backend':self.backend_limit})
        namespace=hashlib.sha256((self.token+self.scope+policy).encode()).hexdigest()[:20]
        self.root=session.root/'ri1'/namespace if session.persistent else None
        self.results=OrderedDict();self.result_bytes=0;self.result_limit=32*1024**2
        self.metrics=session.metrics.setdefault('render_index',{'fresh_candidates':0,'replayed_rejections':0,'accepted_lookups':0,'producer_seconds':0.,'budget_refusals':0})

    def _read(self,key):
        if self.root is None:
            with self.lock:
                if key in self.memory:self.memory.move_to_end(key)
                return deepcopy(self.memory.get(key))
        try:
            doc=json.loads((self.root/(key+'.json')).read_text(encoding='utf-8'))
            if hashlib.sha256(_packed(doc['value']).encode()).hexdigest()!=doc['sha256']:return None
            return doc['value']
        except(OSError,ValueError,KeyError,TypeError):return None

    def _write(self,key,value):
        if self.root is None:
            size=len(_packed(value).encode())
            if size>self.memory_limit and key!='budget':return
            with self.lock:
                if key in self.memory:
                    self.memory_bytes-=self.memory_sizes.pop(key);del self.memory[key]
                while self.memory and self.memory_bytes+size>self.memory_limit:
                    victim=next((k for k in self.memory if k!='budget'),None)
                    if victim is None:break
                    self.memory_bytes-=self.memory_sizes.pop(victim);del self.memory[victim]
                self.memory[key]=deepcopy(value);self.memory_sizes[key]=size;self.memory_bytes+=size
            return
        from .storage import write_json
        write_json(self.root/(key+'.json'),{'value':value,'sha256':hashlib.sha256(_packed(value).encode()).hexdigest()})

    def _group(self,stage,deps):
        if stage=='clip':
            rows=[deps['record']];ids=[r['id']for r in rows]
            sources=[r.get('provenance',{}).get('source_sha256')or r['path']for r in rows]
        else:
            ids=[deps['transition']['from'],deps['transition']['to']]
            sources=[deps[k].get('source_path',deps[k].get('id',ids[i]))for i,k in enumerate(('outgoing','incoming'))]
        from .execution import key_for
        return key_for({'stage':stage,'sources':sources,'engine':self.session.objects.engine_for(stage)}),ids

    @contextmanager
    def _budget_guard(self):
        from .execution import object_lock
        from contextlib import nullcontext
        with self.lock,object_lock(self.root/'budget.lock')if self.root else nullcontext():yield

    def _reserve(self,group,ids,stage):
        state=self._read('budget')or{'groups':{},'extra_candidates':0}
        item=state['groups'].setdefault(group,{'attempts':0,'recovery_seconds':0.,'first_seconds':0.})
        extra=item['attempts']>0
        exhausted=(item['attempts']>=self.limit or(extra and(item['recovery_seconds']>=max(self.seconds,3*item['first_seconds'])or state['extra_candidates']>=self.extra_limit)))
        if exhausted:
            self.metrics['budget_refusals']+=1
            raise RenderSearchBudgetExceeded(dict(stage=stage,source_ids=ids,attempts=item['attempts'],
                recovery_seconds=item['recovery_seconds'],limits={'candidates':self.limit,'programme_extra_candidates':self.extra_limit}))
        item['attempts']+=1
        if extra:state['extra_candidates']+=1
        self._write('budget',state);return extra

    def _finish(self,group,seconds,extra):
        state=self._read('budget');item=state['groups'][group]
        if extra:item['recovery_seconds']+=seconds
        else:item['first_seconds']=seconds
        self._write('budget',state)

    def cached(self,original,stage,dependencies,producer,owned):
        from .execution import key_for,object_lock
        from .render import SourceWindowError
        from .transition_quality import TransitionFailure
        if stage in('phrase_repeat','source_controls','source_references','sync_windows')and not self.session.persistent:
            key=key_for({'stage':stage,'dependencies':dependencies})
            with self.lock:key_lock=self.key_locks.setdefault(key,threading.RLock())
            with key_lock:
                known=self._memory_result(key)
                if known is not None:return known
                value=original(stage,dependencies,producer,owned);self._remember(key,value,metadata_only=stage=='sync_windows');return value
        if stage not in('clip','overlap'):return original(stage,dependencies,producer,owned)
        deps=clip_dependencies(dependencies)if stage=='clip'else dependencies
        key=key_for({'engine':self.session.objects.engine_for(stage),'stage':stage,'dependencies':deps})
        group,ids=self._group(stage,deps)
        from contextlib import nullcontext
        guard=object_lock(self.root/(key+'.lock'))if self.root else nullcontext()
        with self.lock:key_lock=self.key_locks.setdefault(key,threading.RLock())
        with key_lock,guard:
            if not self.session.persistent:
                known=self._memory_result(key)
                if known is not None:
                    self.metrics['accepted_lookups']+=1;return known
            saved=self._read(key)
            if saved and saved.get('state')=='rejected':
                value=saved['error'];evidence=deepcopy(value['evidence'])
                evidence['render_index_replayed_rejection']=True
                self.metrics['replayed_rejections']+=1
                if value['kind']=='source':raise SourceWindowError(value['identity'],value['roles'],value['reason'],evidence)
                raise TransitionFailure(value['outgoing'],value['incoming'],value['reason'],evidence)
            def produce():
                with self._budget_guard():extra=self._reserve(group,ids,stage)
                began=time.monotonic()
                self.metrics['fresh_candidates']+=1
                token=_correction_budget.set({'used':0,'maximum':self.backend_limit})if stage=='clip'else None
                try:return producer()
                except(SourceWindowError,TransitionFailure)as error:
                    if isinstance(error,SourceWindowError):
                        value=dict(kind='source',identity=error.identity,roles=error.roles,
                                   reason=str(error).removeprefix(error.identity+': '),evidence=error.evidence)
                    else:
                        value=dict(kind='transition',outgoing=error.outgoing,incoming=error.incoming,
                                   reason=str(error).removeprefix(error.outgoing+' → '+error.incoming+': '),evidence=error.evidence)
                    try:self._write(key,dict(state='rejected',stage=stage,error=value))
                    except(ValueError,TypeError):pass
                    raise
                finally:
                    if token is not None:_correction_budget.reset(token)
                    elapsed=time.monotonic()-began
                    with self._budget_guard():self._finish(group,elapsed,extra)
                    self.metrics['producer_seconds']+=elapsed
            # Ordinary stage caching still independently checks every artifact.
            # An accepted index entry alone never approves or returns audio.
            value=original(stage,deps,produce,owned)
            if not self.session.persistent:self._remember(key,value)
            if not saved or saved.get('state')!='accepted':self._write(key,dict(state='accepted',stage=stage,source_ids=ids))
            self.metrics['accepted_lookups']+=1
            return value

    def _memory_result(self,key):
        from .execution import sha256
        with self.lock:item=self.results.get(key)
        if item is None:return None
        try:valid=all(sha256(path)==digest for path,digest in item['files'].items())
        except OSError:valid=False
        if valid:
            with self.lock:
                if self.results.get(key)is item:self.results.move_to_end(key)
            return deepcopy(item['value'])
        with self.lock:
            if self.results.get(key)is item:
                self.result_bytes-=item['bytes'];del self.results[key]
        return None

    def _remember(self,key,value,*,metadata_only=False):
        from .execution import sha256
        files={}
        def visit(v):
            if isinstance(v,dict):
                for child in v.values():visit(child)
            elif isinstance(v,(tuple,list)):
                for child in v:visit(child)
            elif isinstance(v,(str,Path)):
                try:
                    p=Path(v).resolve()
                    if p.is_file()and(p.is_relative_to(self.session.work)or p.is_relative_to(self.session.root)):
                        files[str(p)]=sha256(p)
                except(OSError,ValueError):pass
        if not metadata_only:visit(value)
        size=len(_packed(value).encode())
        if size>self.result_limit:return
        with self.lock:
            if key in self.results:self.result_bytes-=self.results.pop(key)['bytes']
            while self.results and self.result_bytes+size>self.result_limit:
                _,old=self.results.popitem(last=False);self.result_bytes-=old['bytes']
            self.results[key]={'value':deepcopy(value),'files':files,'bytes':size};self.result_bytes+=size


def enable(session,scope,**options):
    if not hasattr(session,'render_index'):session.render_index=RenderIndex(session,scope,**options)
    return session.render_index


def indexed_source_call(stage,function,path,warp,sample_rate,*args,**options):
    """Index exact native-clock/template evidence before expensive stretching.

    Backend results and the actual blend still need their independent audits.
    Both the native PCM and complete initial map participate in the identity.
    """
    if stage not in('source_controls','source_references'):raise ValueError('Unknown native evidence stage')
    from .execution import current,cached
    session=current()
    if session is None or not hasattr(session,'render_index'):
        return function(path,warp,sample_rate,*args,**options)
    def produce():
        result=function(path,warp,sample_rate,*args,**options)
        return {'warp':deepcopy(warp),'result':result}
    value=cached(stage,{'pcm':path,'initial_map':warp,'sample_rate':sample_rate,
        'args':args,'options':options,'function':function.__module__+'.'+function.__qualname__},produce,Path(path).parent)
    warp.clear();warp.update(deepcopy(value['warp']))
    return deepcopy(value['result'])


def cached_phrase_repeat(source,destination,edit,*,context_path):
    """Return the exact repeated PCM and native context without copying it again."""
    from .loops import repeat_file
    from .execution import current,cached
    session=current()
    if not session:return str(destination),repeat_file(source,destination,edit,context_path=context_path)
    if not session.persistent and not hasattr(session,'render_index'):
        return str(destination),repeat_file(source,destination,edit,context_path=context_path)
    recipe={k:v for k,v in edit.items()if k!='native_join_contexts'}
    def produce():
        context=repeat_file(source,destination,edit,context_path=context_path)
        return {'path':str(destination),'context':context}
    value=cached('phrase_repeat',{'source':source,'recipe':recipe},produce,Path(destination).parent)
    # Canonical content-checked context references stabilize the old rebasing tag.
    return value['path'],value['context']
