def is_writeable_property(obj: object, var: str):
    ''' Check if possible to set var with current object '''
    if hasattr(obj, var):
        return True
    if not hasattr(obj.__class__, var):
        return False

    attr = getattr(obj.__class__, var)
    return isinstance(attr, property) and attr.fset is not None