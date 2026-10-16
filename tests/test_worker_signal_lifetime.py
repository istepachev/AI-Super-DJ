"""The caller can remove an inference input while a persistent model owns views."""
import io,json,mmap
import numpy as np
import pytest

from superdj import analysis,model_worker


@pytest.mark.parametrize('first_fails',[False,True])
def test_beat_worker_releases_input_before_reply_even_when_model_retains_views(tmp_path,monkeypatch,first_fails):
    paths=[tmp_path/'first.npy',tmp_path/'second.npy']
    expected=[np.arange(1024,dtype='float32'),np.arange(2048,dtype='float32')+.5]
    for path,a in zip(paths,expected):np.save(path,a)
    retained=[];replies=[]
    def tracker(signal,checkpoint,device):
        retained.append(signal[::2])
        if len(retained)==1 and first_fails:raise ValueError('inference rejected')
        return np.array([.1,.6]),np.array([.1]),{}
    monkeypatch.setattr(analysis,'_beat_this',tracker)
    class Parent(io.StringIO):
        def flush(self):
            index=len(replies)
            if index>=len(paths):return
            # A tensor retaining a view must not keep an mmap/file descriptor.
            base=retained[index]
            while getattr(base,'base',None) is not None:base=base.base
            assert not isinstance(base,mmap.mmap)
            moved=paths[index].with_suffix('.released')
            paths[index].rename(moved);moved.unlink()
            np.testing.assert_array_equal(retained[index],expected[index][::2])
            replies.append(json.loads(self.getvalue().splitlines()[-1]))
    requests=''.join(json.dumps({'kind':'beat','job':{'signal':str(p),'checkpoint':'fixture','device':'cpu'}})+'\n' for p in paths)
    monkeypatch.setattr('sys.stdin',io.StringIO(requests))
    monkeypatch.setattr('sys.stdout',Parent())
    model_worker.main()
    assert len(replies)==2 and all(not p.exists() for p in paths)
    assert ('error' in replies[0])==first_fails and 'result' in replies[1]
    np.testing.assert_array_equal(retained[0],expected[0][::2])
