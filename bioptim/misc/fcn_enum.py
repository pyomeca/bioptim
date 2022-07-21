
#
# straight from Python's enum.py
#

def _is_descriptor(obj):  # useless for now since we want functions and functions are descriptors
    """
    Returns True if obj is a descriptor, False otherwise.
    """
    return (
            hasattr(obj, '__get__') or
            hasattr(obj, '__set__') or
            hasattr(obj, '__delete__')
            )

def _is_dunder(name):
    """
    Returns True if a __dunder__ name, False otherwise.
    """
    return (
            len(name) > 4 and
            name[:2] == name[-2:] == '__' and
            name[2] != '_' and
            name[-3] != '_'
            )

def _is_sunder(name):
    """
    Returns True if a _sunder_ name, False otherwise.
    """
    return (
            len(name) > 2 and
            name[0] == name[-1] == '_' and
            name[1:2] != '_' and
            name[-2:-1] != '_'
            )

def _is_private(cls_name, name):
    # do not use `re` as `re` imports `enum`
    pattern = '_%s__' % (cls_name, )
    pat_len = len(pattern)
    if (
            len(name) > pat_len
            and name.startswith(pattern)
            and name[pat_len:pat_len+1] != ['_']
            and (name[-1] != '_' or name[-2] != '_')
        ):
        return True
    else:
        return False

def _is_valid_member(cls_name, name):
    return not (_is_descriptor(name)
        or _is_sunder(name)
        or _is_dunder(name)
        or _is_private(cls_name, name))


#
# by me inspired by Python's enum.py
#
# was created for a very specific goal and not general in the current form
#
class MetaFcnEnum(type):

    def __new__(metacls, cls, bases, classdict):

        fcn_names = metacls._find_fcn_names(cls, classdict)
        fcn_class = super().__new__(metacls, cls, bases, classdict)

        for fcn_name in fcn_names:
            fcn = object.__new__(fcn_class)  # no fancy __new__ stuff
            fcn.__init__(fcn_name, classdict[fcn_name])
            setattr(fcn_class, fcn_name, fcn)

        return fcn_class

    def __getattribute__(cls, attr):  # hides bases' members at runtime, but not in PyCharm
                                      # TODO: crashes hasattr when it should simply return False
                                      # maybe add __getattr__ and __hasattr__ to fix. Test with just __getattr__ too.
        getattribute = object.__getattribute__

        classdict = getattribute(cls, '__dict__')
        classname = getattribute(cls, '__name__')
        obj = classdict[attr] if attr in classdict else None

        is_valid_member = not (_is_descriptor(obj)
                               or _is_dunder(attr)
                               or _is_sunder(attr)
                               or _is_private(classname, attr))

        if is_valid_member and obj is None:
            raise TypeError(f"'{classname}' has no member '{attr}'.")
        if is_valid_member and obj is not None:
            return classdict[attr]
        else:
            return getattribute(cls, attr)

    @staticmethod
    def _find_fcn_names(classname, classdict):
        return tuple( filter(lambda key: _is_valid_member(classname, key), classdict.keys()) )

# TODO: differentiate a method from a member (with a decorator?).

class FcnEnum(metaclass=MetaFcnEnum):

    def __init__(self, name, fcn):
        self.name = name
        self.fcn = fcn

    def __call__(self, *args, **kwargs):  # allows to call Fcn member directly
        return self.fcn(*args, **kwargs)

    def __str__(self):
        return f"{self.__class__.__name__}.{self.name}: {self.fcn}"

breakpoint()