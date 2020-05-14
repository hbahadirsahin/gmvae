from utils.logger import Logger


class Step(Logger):
    """Step to be run by Workflow. Is functional so must return something.

    On initialization will only accept one attribute, element, that will be
    operated on by the step.

    The method with the name defined by INSERTION_POINT:
        * Must be created
        * Must return the result from running the Step.

    Attributes:
        _workflow (Workflow): Reference to Workflow instance to access common
            data.

    Class Attributes:
        INSERTION_POINT (str): Name of the method to call so the step
            operates.
        GRACEFUL (bool): Boolean that determines if the workflow should stop
            running steps in the event a step throws an exception. True means
            the workflow will continue.

    Class Properties:
        logger (Logger): Class property that returns a logger instance
            connected to standard out.
    """
    INSERTION_POINT = 'run'
    GRACEFUL = False

    def __init__(self, workflow=None):
        self._workflow = workflow

    def run(self, element):
        """Insertion point for Step.

        This method should be rewritten and expanded upon in all classes that
        inherit from Step.

        Args:
            element (object): Object step should operate on.

        Returns:
            element (object): Unmodified object in this case.
        """
        return element
