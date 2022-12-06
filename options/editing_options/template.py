
# This config saves inversion and smile edit results

edit_configs = [
    dict( method='inversion'),
    dict( method='interfacegan', edit='smile', strength=2),
    dict( method='ganspace', edit='overexposed',   strength=5),
    dict( method='styleclip', type='mapper', edit='purple_hair', strength=0.08),
]
