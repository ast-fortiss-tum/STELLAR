def merged_callbacks(*callbacks):
    def callback(*args, **kwargs):
        for c in callbacks:
            c(*args, **kwargs)
    return callback
