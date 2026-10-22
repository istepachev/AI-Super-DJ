"""Factory CUE pacing, distinct from full-deck and solo durations."""
import json
from copy import deepcopy

import numpy as np
import pytest
import soundfile as sf

from superdj.cli import main
from superdj.compact import arrange_sync
from superdj.minimum import preserve_minimum
from superdj.minimum_timeline import native_plan
from superdj.planner import plan_tracks
from superdj.render import render_plan, RenderError
from test_new_workflow import record


@pytest.mark.parametrize('style', ['zhilin', 'sync'])
@pytest.mark.parametrize('mode', ['mood', 'classic', 'fuzzy'])
def test_factory_cli_targets_cue_intervals_in_every_style_and_mode(tmp_path, monkeypatch, style, mode):
    monkeypatch.setattr('superdj.cues.runtime_config', lambda: {})
    rows=[record(c, 480) for c in 'abc']
    source=tmp_path/'records.json'
    source.write_text(json.dumps(rows), encoding='utf8')
    main(['plan','--records',str(source),'--work',str(tmp_path/'work'),
          '--mode','fixed','--style',style,'--harmonic-mode',mode])
    planned=json.loads((tmp_path/'work/plan.json').read_text(encoding='utf8'))
    assert planned['order']==['a','b','c']
    assert all(e['overlap_bars']==32 for e in planned['edges'])
    for section in planned['sections'].values():
        assert 180<=section['predicted_chapter_seconds']<=240
        assert section['predicted_play_seconds']<=300
    middle=planned['sections']['b']
    assert middle['predicted_play_seconds']>middle['predicted_chapter_seconds']+60


def test_optional_loop_uses_short_chapter_even_when_full_deck_exceeds_two_minutes(tmp_path):
    from test_minimum_stem_alignment import inputs
    _,rows,planned,prepared=inputs(tmp_path)
    section=planned['sections']['a']
    section.update(duration_budget_basis='chapter interval',predicted_chapter_seconds=100.,
                   predicted_play_seconds=125.)
    original=deepcopy(planned)
    extended,_,_=preserve_minimum(planned,rows,prepared,tmp_path/'repeat')
    edit=extended['minimum_playback_edits'][0]
    assert extended['sections']['a']['predicted_chapter_seconds']==pytest.approx(100+edit['added_seconds'])
    assert extended['sections']['a']['predicted_play_seconds']==pytest.approx(125+edit['added_seconds'])
    assert extended['sections']['a']['predicted_play_seconds']<=300
    restored=native_plan(extended)
    assert restored['sections']['a']['predicted_chapter_seconds']==100
    assert restored['sections']['a']['predicted_play_seconds']==pytest.approx(125,abs=1e-12)
    assert planned==original


def test_110_second_chapter_is_retained_when_a_clean_loop_is_unavailable(tmp_path,monkeypatch):
    from test_minimum_stem_alignment import inputs
    _,rows,planned,prepared=inputs(tmp_path)
    planned['sections']['a'].update(duration_budget_basis='chapter interval',
        predicted_chapter_seconds=110.,predicted_play_seconds=140.)
    monkeypatch.setattr('superdj.short_tracks.choose_repeat',lambda *a,**k:(None,'native melody conflicts'))
    result,_,files=preserve_minimum(planned,rows,prepared,tmp_path/'repeat')
    assert not result['minimum_playback_edits']
    assert result['short_track_loop_reviews'][0]['short_play_retained']
    assert result['sections']['a']['predicted_chapter_seconds']==110
    assert files==prepared


def test_real_native_pcm_has_the_planned_final_cue_chapter(tmp_path):
    # A complete real PCM assembly, without model or backend substitutes. A
    # one-track chapter ends at the set end rather than a nonexistent next cue.
    sr=8000;row=record('one',320)
    t=np.arange(320*sr)/sr
    source=tmp_path/'native.wav'
    audio=np.column_stack([.06*np.sin(2*np.pi*80*t)+.03*np.sin(2*np.pi*400*t)]*2)
    sf.write(source,audio,sr,subtype='FLOAT');row['path']=str(source)
    planned=plan_tracks([row],mode='fixed')
    planned['mixing_preferences']={'duration_budget_basis':'chapter interval'}
    planned=arrange_sync(planned,[row],sample_rate=sr)
    rendered=render_plan(planned,[row],{},tmp_path/'mix.wav',tmp_path/'render',sample_rate=sr)
    assert rendered['assembly_verification']['verified']
    actual=rendered['tracks'][0]
    assert 180<=actual['chapter_seconds']<=240
    assert actual['chapter_seconds']==pytest.approx(sf.info(tmp_path/'mix.wav').frames/sr)
    assert actual['chapter_seconds']==pytest.approx(planned['sections']['one']['predicted_chapter_seconds'])
    inconsistent=deepcopy(planned)
    inconsistent['sections']['one']['predicted_chapter_seconds']-=60
    with pytest.raises(RenderError,match='actual CUE chapter disagrees'):
        render_plan(inconsistent,[row],{},tmp_path/'rejected.wav',tmp_path/'bad-render',sample_rate=sr)
    assert not (tmp_path/'rejected.wav').exists()
