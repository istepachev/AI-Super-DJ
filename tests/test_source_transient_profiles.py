from superdj.stretch_backend import backend_arguments
import pytest

def test_ordinary_backend_options_unchanged():
    assert backend_arguments('r2')==['--fast','--detector-perc']
    assert backend_arguments('r3')==['--fine']

@pytest.mark.parametrize('profile,options',[('short',['--window-short']),('crisp6',['--crisp','6'])])
def test_pitch_preserving_profiles_are_explicit_bounded_source_choices(profile,options):
    assert backend_arguments('r2',profile)==['--fast','--detector-perc',*options]

@pytest.mark.parametrize('engine,profile',[('r3','short'),('r3','crisp6'),('r2','anything')])
def test_unknown_or_inapplicable_profile_is_refused(engine,profile):
    with pytest.raises(ValueError):backend_arguments(engine,profile)
