import pytest

from superdj.harmonize_calibration import fit_successor_controls, cross_validate_successor_controls
from superdj.harmonize_successors import evaluate_successors


def example(name='x', prefix=''):
    return dict(name=name, records=[dict(id=prefix+i, bpm=bpm, key=key) for i,bpm,key in
        [('a',120,'8A'),('b',122,'9A'),('c',140,'8A'),('d',124,'10A')]],
        order=[prefix+i for i in ['a','b','d','c']])


def test_fit_counts_match_real_ranker_and_does_not_install_default():
    e=example()
    controls=[dict(mode='fuzzy',priority=p,direction='ascending') for p in (0,.5,1)]
    fit=fit_successor_controls([e],candidates=controls,min_candidates=1)
    actual=evaluate_successors(e['records'],e['order'],**fit['settings'])['summary']['all']
    assert fit['training']['edge_count']==actual['edge_count']
    assert fit['training']['best_count']==actual['best_count']
    assert fit['training']['mean_midpoint_reciprocal_rank']==pytest.approx(actual['mean_midpoint_reciprocal_rank'])
    assert fit['installs_default'] is False


def test_tied_rank_metric_is_not_confident_top1():
    e=dict(name='ties',records=[dict(id=i,bpm=120,key='8A') for i in 'abcd'],order=list('abcd'))
    fit=fit_successor_controls([e], candidates=[{}], min_candidates=2)
    assert fit['training']['best_count']==2
    assert fit['training']['fractional_top1_fraction']==pytest.approx((1/3+1/2)/2)
    assert fit['training']['mean_midpoint_reciprocal_rank']==pytest.approx((1/2+1/1.5)/2)


def test_holdout_is_not_used_to_select_controls():
    examples=[example('one','1'),example('two','2'),example('three','3')]
    controls=[dict(priority=p) for p in (0,.5,1)]
    cv=cross_validate_successor_controls(examples,candidates=controls,min_candidates=1)
    for i,fold in enumerate(cv['folds']):
        train=[e for j,e in enumerate(examples) if i!=j]
        independent=fit_successor_controls(train,candidates=controls,min_candidates=1)
        assert fold['fitted']['settings']==independent['settings']
        assert fold['previously_seen_directed_pairs']==0
    assert not cv['defaults_changed']


@pytest.mark.parametrize('minimum',[0,-1,True,2.5])
def test_invalid_minimum_rejected(minimum):
    with pytest.raises(ValueError):fit_successor_controls([example()],min_candidates=minimum)


def test_insufficient_data_rejected_and_duplicate_examples_rejected():
    with pytest.raises(ValueError,match='enough'):fit_successor_controls([example()])
    with pytest.raises(ValueError,match='unique'):cross_validate_successor_controls([example(),example()],min_candidates=1)
    with pytest.raises(ValueError,match='two'):cross_validate_successor_controls([example()])


def test_renaming_tracks_changes_no_fitted_controls():
    settings=[dict(priority=p) for p in (0,.5,1)]
    a=fit_successor_controls([example(prefix='original')],candidates=settings,min_candidates=1)
    b=fit_successor_controls([example(prefix='different')],candidates=settings,min_candidates=1)
    assert a['settings']==b['settings'] and a['training']==b['training']


def test_near_ties_are_anchored_not_transitive_during_fitting():
    e=dict(name='near',records=[dict(id=i,bpm=bpm,key='8A') for i,bpm in
        [('current',120),('actual',125+8e-9),('x',125),('y',125+4e-9)]],
        order=['current','actual','x','y'])
    fit=fit_successor_controls([e],candidates=[dict(priority=0)],min_candidates=3)
    audit=evaluate_successors(e['records'],e['order'],**fit['settings'])
    assert audit['edges'][0]['actual_rank']==3
    assert fit['training']['mean_midpoint_reciprocal_rank']==pytest.approx(1/3)


def test_cohort_variants_are_never_used_to_fit_their_heldout_group():
    a = dict(example('one', '1'), cohort_group='a')
    a2 = dict(example('one-alternative', '1'), cohort_group='a')
    a2['order'] = a2['order'][::-1]
    b = dict(example('two', '2'), cohort_group='b')
    cv = cross_validate_successor_controls([a, a2, b], candidates=[{}], min_candidates=1)
    assert cv['validation'] == 'leave-one-cohort-out'
    for fold in cv['folds']:
        expected = ['two'] if fold['heldout_cohort_group'] == 'a' else ['one', 'one-alternative']
        assert fold['training_example_names'] == expected
        assert fold['previously_seen_directed_pairs'] == 0
    with pytest.raises(ValueError, match='two independent'):
        cross_validate_successor_controls([a, a2], candidates=[{}], min_candidates=1)
