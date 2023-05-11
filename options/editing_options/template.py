
# This config saves inversion and smile edit results

edit_configs = [
    dict( method='inversion'),
    dict( method='interfacegan', edit='smile', strength=2),
    dict( method='ganspace', edit='overexposed', strength=5),
    dict( method='styleclip', type='mapper', edit='purple_hair', strength=0.08),
    dict( method='styleclip', type='global', neutral_text='a face', target_text='a face with glasses', strength=4, disentanglement=0.11),
    dict( method='gradctrl', num_steps=9, learning_rate=0.5, direction='negative', edit="gender"),
] 
