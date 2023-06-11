import numpy as np


def default_transform(x):
    """Default parameter transformation, which does nothing."""
    return x


class BaseModel:
    """Base class to create the model and contain common functions.

    Parameters
    ----------
    N: int
        Number of parameters in the model.
    M: int
        Number of predictions that the model makes.
    data: np.ndarray (M,) (optional)
        Data values for each independent variable. The data is needed when evaluating
        the residual and cost.
    data_error: np.ndarray (M,) (optional)
        Error bar for each data value. This is required when evaluating the residual and
        cost.
    transform: callable ``f(x)`` (optional)
        A function to transform the parameters from whatever space the parameters are
        input to the parameter spaces used by the model.
    """

    def __init__(
        self,
        N: int,
        M: int,
        data: np.ndarray = None,
        data_error: np.ndarray = None,
        transform=None,
    ):
        # Dimensionality
        self.N = N
        self.M = M

        self.data = data
        self.data_error = 1.0 if data_error is None else data_error

        # Parameter transform
        if transform is None:
            self.transform = default_transform
        else:
            self.transform = transform

    def predict(self, params):
        """Evaluate the model at the given parameters."""
        raise ModelError("Model prediction routine hasn't been implemented")

    def residual(self, params: np.ndarray) -> np.ndarray:
        """Evaluate the residual of the model at the given parameter values.

        Parameters
        ----------
        params : np.ndarray
            Parameter values to evaluate.

        Returns
        -------
        np.ndarray
            Residual of the model.

        """
        check_data(self.data, self.data_error)
        preds = self.predict(params)
        return (self.data - preds) / self.data_error

    def cost(self, params: np.ndarray) -> float:
        """Evaluate the weighted least squares cost at the given parameter values.

        Parameters
        ----------
        params : np.ndarray
            Parameter values to evaluate.

        Returns
        -------
        float
            Cost value.

        """
        check_data(self.data, self.data_error)
        res = self.residual(params)
        return 0.5 * np.linalg.norm(res) ** 2


class LinearModel(BaseModel):
    """A model class for a linear model with monomial basis function.

    The linear model with :math:`N` parameters has the form

    .. math::

       f(\theta; t) = \sum_{n=0}^{N-1} \theta_n t^n = J \vec{\theta},

    where :math:`J` is the design matrix.

    Parameters
    ----------
    N: int
        Nummber of parameters in the model.
    t: np.ndarray (M,)
        List of values for the independent variables.
    data: np.ndarray (M,) (optional)
        Data values for each independent variable. The data is needed when evaluating
        the residual and cost.
    data_error: np.ndarray (M,) (optional)
        Error bar for each data value. This is required when evaluating the residual and
        cost.
    transform: callable ``f(x)`` (optional)
        A function to transform the parameters from whatever space the parameters are
        input to the parameter spaces used by the model.
    """

    def __init__(
        self,
        N: int,
        t: np.ndarray,
        data: np.ndarray = None,
        data_error: np.ndarray = None,
        transform=None,
    ):
        self.t = t
        super().__init__(N, len(t), data, data_error, transform)

        # Design matrix
        self.J = np.array([t**n for n in range(self.N)]).T

    def predict(self, params: np.ndarray) -> np.ndarray:
        """Evaluate the model at the given parameters.

        Parameters
        ----------
        params : np.ndarray (N,)
            Parameter values to evaluate.

        Returns
        -------
        np.ndarray
            Predictions of the model.

        """
        x = self.transform(params)
        return self.J @ x


class FractionalModel(BaseModel):
    """A model class for fractional models.

    The model has the form

    .. math::

       f(\theta; t) = \frac{1}{\sum_{n=0}^{N-1} \theta_n t^n + t^n}

    Parameters
    ----------
    N: int
        Nummber of parameters in the model.
    t: np.ndarray (M,)
        List of values for the independent variables.
    data: np.ndarray (M,) (optional)
        Data values for each independent variable. The data is needed when evaluating
        the residual and cost.
    data_error: np.ndarray (M,) (optional)
        Error bar for each data value. This is required when evaluating the residual and
        cost.
    transform: callable ``f(x)`` (optional)
        A function to transform the parameters from whatever space the parameters are
        input to the parameter spaces used by the model.
    """

    def __init__(
        self,
        N: int,
        t: np.ndarray,
        data: np.ndarray = None,
        data_error: np.ndarray = None,
        transform=None,
    ):
        self.t = t
        super().__init__(N, len(t), data, data_error, transform)

    def predict(self, params):
        """Evaluate the model at the given parameters.

        Parameters
        ----------
        params : np.ndarray (N,)
            Parameter values to evaluate.

        Returns
        -------
        np.ndarray
            Predictions of the model.

        """
        x = self.transform(params)
        denom_elems = np.array([xi * self.t**ii for ii, xi in enumerate(x)])
        return 1 / (np.sum(denom_elems, axis=0) + self.t**self.N)


class ExponentialModel(BaseModel):
    """A model class for a model of the sum of decaying expponentials.

    The model with :math:`N` parameters has the form

    .. math::

       f(\theta; t) = \frac{1}{N} \sum_{n=0}^{N-1} \exp \left( -\theta_n t \right).


    Parameters
    ----------
    N: int
        Nummber of parameters in the model.
    t: np.ndarray (M,)
        List of values for the independent variables.
    data: np.ndarray (M,) (optional)
        Data values for each independent variable. The data is needed when evaluating
        the residual and cost.
    data_error: np.ndarray (M,) (optional)
        Error bar for each data value. This is required when evaluating the residual and
        cost.
    transform: callable ``f(x)`` (optional)
        A function to transform the parameters from whatever space the parameters are
        input to the parameter spaces used by the model.
    """

    def __init__(
        self,
        N: int,
        t: np.ndarray,
        data: np.ndarray = None,
        data_error: np.ndarray = None,
        transform=None,
    ):
        self.t = t
        super().__init__(N, len(t), data, data_error, transform)

    def predict(self, params):
        """Evaluate the model at the given parameters.

        Parameters
        ----------
        params : np.ndarray (N,)
            Parameter values to evaluate.

        Returns
        -------
        np.ndarray
            Predictions of the model.

        """
        x = self.transform(params)
        terms = np.array([np.exp(-par * self.t) for par in x])
        return np.sum(terms, axis=0) / self.N


def check_data(data, error):
    """Check if the data and the corresponding error bars are set."""
    if data is None:
        raise ModelError("Please set the values and error bars of the data")


class ModelError(Exception):
    def __init__(self, msg):
        super().__init__(msg)
        self.msg = msg
