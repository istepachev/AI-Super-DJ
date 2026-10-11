"""Actual learning/profile/remaining-candidate CLI workflows on synthetic files."""
import json

import pytest

from superdj.cli import main
from superdj.harmonize_ranker import feature_names, MODEL_VERSION, UTILITY_WARNING


def dump(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')
    return str(path)


def profile():
    names = feature_names('circle')
    weights = [0.] * len(names)
    weights[names.index('abs_bpm_delta/10')] = -1.
    return dict(schema_version=1, model_version=MODEL_VERSION, experimental=True,
                feature_family='circle', feature_names=names, coefficients=weights,
                l2_penalty=1., min_candidates=2, identifier_features=False,
                default_changed=False, utility_warning=UTILITY_WARNING,
                training_provenance=[dict(name='one', metadata_sha256='a'*64),
                                     dict(name='two', metadata_sha256='b'*64)])


def test_rank_cli_excludes_played_ids_and_keeps_blocked_pair_visible(tmp_path):
    rows = [dict(id=i, bpm=b, key=k) for i,b,k in
            [('a', 120, '8A'), ('b', 125, '9A'), ('c', 126, '10A'), ('d', 125, '3B')]]
    args = ['harmonize-rank', '--records', dump(tmp_path/'records.json', rows),
            '--profile', dump(tmp_path/'profile.json', profile()), '--current', 'b',
            '--played-prefix', dump(tmp_path/'played.json', ['a','b']),
            '--work', str(tmp_path/'ranks')]
    main(args)
    result = json.loads((tmp_path/'ranks/next_track_rankings.json').read_text(encoding='utf-8'))
    assert result['remaining_ids'] == ['c','d']
    assert result['best_candidate_ids'] == ['c']
    assert result['blocked_count'] == 1
    assert next(row for row in result['candidates'] if row['id']=='d')['rank'] is None
    assert result['whole_path_optimized'] is False


@pytest.mark.parametrize('prefix', [['b','a'], ['a','a','b'], ['missing','b'], 'b'])
def test_rank_cli_rejects_invalid_played_prefix(tmp_path, prefix):
    records = dump(tmp_path/'records.json', [dict(id='a',bpm=120,key='8A'), dict(id='b',bpm=122,key='9A')])
    with pytest.raises(ValueError, match='Played prefix'):
        main(['harmonize-rank','--records',records,'--profile',dump(tmp_path/'profile.json',profile()),
              '--current','b','--played-prefix',dump(tmp_path/'played.json',prefix),
              '--work',str(tmp_path/'ranks')])


def test_ranker_learning_cli_writes_nested_validation_and_private_json_profile(tmp_path):
    rows = [dict(id=str(i),bpm=120+i,key='8A') for i in range(6)]
    source = dump(tmp_path/'records.json', rows)
    manifest = dump(tmp_path/'examples.json',
                    [dict(name=str(n),records='records.json',order=[str(i) for i in range(6)]) for n in range(3)])
    main(['harmonize-learn','--learner','ranker','--examples',manifest,
          '--work',str(tmp_path/'learn'),'--min-candidates','2'])
    result = json.loads((tmp_path/'learn/successor_ranker_validation.json').read_text(encoding='utf-8'))
    model = json.loads((tmp_path/'learn/successor_ranker_profile.json').read_text(encoding='utf-8'))
    assert len(result['folds']) == 3
    for fold in result['folds']:
        assert fold['heldout'] not in {row['name'] for row in fold['profile']['training_provenance']}
        assert 'raw_local_audit' in fold and 'gated_recommendation' in fold
    assert len(model['training_provenance']) == 3
    assert model['default_changed'] is False
    assert not (tmp_path/'learn/successor_calibration.json').exists()
