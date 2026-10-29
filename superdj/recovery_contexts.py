"""Bounded private memory for custom rebuilds after actual local-quality refusals.

Records and plans use the ordinary JSON manifest schema. The caller supplies a
dependency token covering engine, preparation and private processing inputs.
This is a search aid, never evidence of an intrinsic source defect or a pass.
"""
from copy import deepcopy
import hashlib
import json
import math


def _record(row):
    digest = row.get('provenance', {}).get('source_sha256')
    if not isinstance(digest, str) or len(digest) != 64:
        return None
    return hashlib.sha256(json.dumps(row, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _context(plan, records, identity, dependency_token):
    rows = {r['id']: r for r in records}
    source = rows.get(identity)
    if source is None or _record(source) is None:
        return None
    section = plan['sections'][identity]
    result = {'source': _record(source), 'dependencies': dependency_token,
        'style': plan.get('style'), 'mixing_preferences': deepcopy(plan.get('mixing_preferences', {})),
        'long_play_explicit': plan.get('long_play_explicit'),
        'source_locks': [section.get('locked_start', False), section.get('locked_end', False)],
        'neighbours': []}
    for edge in plan['edges']:
        if identity not in (edge['from'], edge['to']):
            continue
        head = edge['to'] == identity
        other = edge['from'] if head else edge['to']
        fingerprint = _record(rows[other])
        if fingerprint is None:
            return None
        controls = deepcopy(edge)
        # The selected source span is stored separately, not part of the
        # family used to query its previously rejected complete spans.
        controls.setdefault('cue_locks', {}).pop('incoming_start_seconds' if head else 'outgoing_end_seconds', None)
        result['neighbours'].append({'record': fingerprint, 'edge': controls,
            'section': [plan['sections'][other][b] for b in ('start', 'end')]})
    return result


class QualityContexts:
    def __init__(self, document=None, *, dependency_token, maximum_entries=256):
        if not isinstance(dependency_token, str) or not dependency_token:
            raise ValueError('Quality retry memory requires an explicit processing dependency token')
        if isinstance(maximum_entries, bool) or not isinstance(maximum_entries, int) or not 1 <= maximum_entries <= 4096:
            raise ValueError('Quality retry memory must have a bounded entry count')
        self.dependency_token = dependency_token
        self.maximum_entries = maximum_entries
        self.document = deepcopy(document) if document is not None else {'schema_version': 1, 'items': [],
            'native_recording_blacklist': False, 'timing_and_quality_gates_unchanged': True}
        if self.document.get('schema_version') != 1 or not isinstance(self.document.get('items'), list):
            raise ValueError('Unsupported quality retry memory')
        self.document['items'] = self.document['items'][-maximum_entries:]

    def note(self, plan, records, evidence, error):
        local = evidence.get('local_rhythm_coverage', {})
        if local.get('passed') is not False:
            return 0
        added = 0
        for deck in local.get('decks', []):
            identity = deck.get('id')
            if deck.get('sufficient_for_automatic_window') is not False or identity not in plan['sections']:
                continue
            family = _context(plan, records, identity, self.dependency_token)
            if family is None:
                continue
            span = [plan['sections'][identity][b] for b in ('start', 'end')]
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in span) or not 0 <= span[0] < span[1]:
                raise ValueError('Invalid rejected source section')
            if any(item['family'] == family and item['section'] == span for item in self.document['items']):
                continue
            self.document['items'].append({'family': family, 'section': span,
                'failed_role': deck.get('role'), 'error': str(error), 'actual_quality_passed': False})
            self.document['items'] = self.document['items'][-self.maximum_entries:]
            added += 1
        return added

    def exclusions(self, plan, records, identity):
        family = _context(plan, records, identity, self.dependency_token)
        spans = [item['section'] for item in self.document['items'] if family is not None and item['family'] == family]
        return {identity: deepcopy(spans)} if spans else {}


def merge_exclusions(*values):
    """Retain independently established native constraints alongside contexts."""
    result = {}
    for value in values:
        for identity, spans in value.items():
            for span in spans:
                if span not in result.setdefault(identity, []):
                    result[identity].append(list(span))
    return result
