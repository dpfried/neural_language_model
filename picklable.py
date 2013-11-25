import theano
import config

class Picklable(object):
    """
    Rather than pickle the entire model, with theano functions, provide a class
    that specifies which attributes should be pickled. Also avoids issues with
    unpickling Theano objects on a normal machine that were pickled on a GPU,
    by unpacking values from shared objects and saving those
    """
    def _nonshared_attrs(self):
        # should be overridden by subclasses to return a list of strings, which
        # will be the names of object attributes that should be pickled
        return []

    def _shared_attrs(self):
        # should be overridden by subclasses to return a list of strings, which
        # will be the names of theano shared variable object attributes that
        # should be pickled
        return []

    def _initialize(self):
        # override this with any code that should be run after unpickling, e.g.
        # code that relies on the read parameters
        pass

    def _set_attrs(self, **kwargs):
        """
        this method is called with a dictionary of attributes upon object creation
        or reading from a file. Each attribute that is in the dictionary and also
        in the name list returned by _nonshared_attrs or _shared_attrs will be set
        as an attribute of the object.
        """
        for param in self._shared_attrs():
            if type(param) is tuple:
                name, default = param
            else:
                name, default = param, None
            try:
                if name not in kwargs:
                    print 'warning: %s not found, setting to default %s' % (name, default)
                setattr(self, name, theano.shared(kwargs.get(name, default), name=name))
            except TypeError as e: # in case we stored the shared variable, get its current value
                print e
                if name not in kwargs:
                    print 'warning: %s not found, setting to default %s' % (name, default)
                setattr(self, name, theano.shared(kwargs.get(name, default).get_value(), name=name))
        for param in self._nonshared_attrs():
            if type(param) is tuple:
                name, default = param
            else:
                name, default = param, None
            if name not in kwargs:
                print 'warning: %s not found, setting to default %s' % (name, default)
            setattr(self, name, kwargs.get(name, default))

    def __setstate__(self, state):
        """
        called when the object is read from a pickle file. State is a dictionary of attributes,
        the same as is returned by __getstate__
        """
        self._set_attrs(**state)
        if config.DYNAMIC['compile_on_load']:
            self._initialize()

    def __getstate__(self):
        """
        called upon writing the object to a pickle. Returns a dictionary of attributes that are named
        by the _nonshared_attrs and _shared_attrs functions
        """
        state = {}
        for val in self._nonshared_attrs():
            name = val[0] if type(val) is tuple else val
            state[name] = getattr(self, name)
        for val in self._shared_attrs():
            name = val[0] if type(val) is tuple else val
            state[name] = getattr(self, name).get_value()
        return state


