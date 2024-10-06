prompt_type = 'sum'  # ['eol', 'sth', 'sum']
backbone = 'mistral-7b'  # ['opt-6.7b', 'llama-7b', 'llama2-7b', 'mistral-7b']

if prompt_type == 'eol':
    manual_template = 'This sentence : "*sent_0*" means in one word:"'
elif prompt_type == 'sth':
    manual_template = 'This sentence : "*sent_0*" means something'
elif prompt_type == 'sum':
    manual_template = 'This sentence : "*sent_0*" can be summarized as'