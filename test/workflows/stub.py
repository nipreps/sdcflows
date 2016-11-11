'''  Module for creating stubs to aid unit testing '''

def settings(custom_settings=None):
    ''' returns a minimal settings object '''
    stub = custom_settings if custom_settings else {}
    if 'output_dir' not in stub:
        stub['output_dir'] = '.'
    return stub
