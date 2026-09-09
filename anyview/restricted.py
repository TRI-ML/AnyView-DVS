# Shared helper of the restricted-split extractors (restricted_egoexo4d.py, restricted_waymo.py).


def compare_sha256(produced, expected):
    '''
    Names whose sha256 differs (or is missing) between produced and expected {key: {name: sha}}.
    '''
    mismatches = []
    for key, exp in expected.items():
        got = produced.get(key, {})
        for name, sha in exp.items():
            if got.get(name) != sha:
                mismatches.append(f'{key}/{name}')
    return mismatches
