"""
mcup.py
====================================
The core module of MCUP package.
"""


class Measurement:
    """An example docstring for a class definition."""

    def __init__(self, x=None, y=None, x_err=None, y_err=None):
        """
        Blah blah blah.
        Parameters
        ---------
        name
            A string to assign to the `name` instance attribute.
        """
        pass

    def set_data(self, x=None, y=None, x_err=None, y_err=None):
        """[summary]

        Args:
            x ([type], optional): [description]. Defaults to None.
            y ([type], optional): [description]. Defaults to None.
            x_err ([type], optional): [description]. Defaults to None.
            y_err ([type], optional): [description]. Defaults to None.

        Raises:
            AssertionError: [description]
            AssertionError: [description]
        """
        self.x = copy.deepcopy(x)
        self.y = copy.deepcopy(y)
        self.x_err = copy.deepcopy(x_err)
        self.y_err = copy.deepcopy(y_err)

        params = [self.x, self.y, self.x_err, self.y_err]
        if None in params:
            raise AssertionError(
                "To set Measurement data x, y, x_err, y_err have to be set."
            )
        if not all(lambda x: isinstance(x, (list, np.ndarray)), params):
            raise TypeError("All argument have to be list or np.ndarray.")

        for i in range(4):
            if isinstance(params[i], list):
                params[i] = np.array(params[i])
            if np.ndim(params[i]) != 1:
                raise TypeError("All argument have to have ndim=1.")

        self.data_len = self.x.shape[0]

        for i in range(4):
            if np.ndim(params[i].shape[0]) != self.data_len:
                raise TypeError("All argument have to have same size.")

    def about_self(self):
        """
        Return information about an instance created from ExampleClass.
        """
        return "I am a very smart {} object.".format(self.name)
