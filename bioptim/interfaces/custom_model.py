class CustomModel:
    """
    This class is made for the user to help him create his own model
    """

    # ---- absolutely needed to be implemented ---- #
    @property
    def nb_quaternions(self):
        """Number of quaternion in the model"""
        return 0
