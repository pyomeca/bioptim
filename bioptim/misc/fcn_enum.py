
#
# straight from Python's enum.py
#

def _is_descriptor(obj):
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

#        def getisolated(cls, attr):
#            breakpoint()
#            getattribute = object.__getattribute__
#            if _is_valid_member(getattribute(cls, '__name__'), attr) and attr not in getattribute(cls, '__dict__'):
#                raise TypeError(f"{getattribute(cls, '__name__')} does not have member {attr}.")
#            return getattribute(cls, attr)

#        fcn_class.__getattribute__ = getisolated  # TODO: find a way to isolate bases' attributes from child's

        return fcn_class

    @staticmethod
    def _find_fcn_names(classname, classdict):
        return tuple( filter(lambda key: _is_valid_member(classname, key), classdict.keys()) )

# TODO: isolate members of bases from child and differentiate a method from a member (with a decorator?).

class FcnEnum(metaclass=MetaFcnEnum):

    def __init__(self, name, fcn):
        self.name = name
        self.fcn = fcn

    def __call__(self, *args, **kwargs):  # allows to call Fcn member directly
        return self.fcn(*args, **kwargs)

    def __str__(self):
        return f"{self.__class__.__name__}.{self.name}: {self.fcn}"
