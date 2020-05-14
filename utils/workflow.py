import copy

from utils.logger import Logger


class Workflow(Logger):
    """Run steps in sequence for each element in a list of elements.

    Attributes:
        _steps (list of Step): List of callable Step objects that accept a
            single element and perform their operations. If one of the steps is
            None or the given element is None, it will be skipped.
        _pipeline (Pipeline): Reference to Pipeline instance to access common
            data.
        **kwargs (dict of attributes): Attributes to set during initialization.
            Allows for easy way to add data to be accessible through workflow
            reference by steps.

    Class Attributes:
        GRACEFUL (bool): Boolean that determines if the workflow should stop
            running elements in the event a step throws an exception. True
            means the workflow will continue.

    Class Properties:
        logger (Logger): Class property that returns a logger instance connected
            to standard out.
    """
    GRACEFUL = False

    def __init__(self, steps=tuple(), pipeline=None, **kwargs):
        self._steps = steps
        self._pipeline = pipeline

        # Set additional instance attributes that each Step can access by
        # calling self._workflow.
        for key, value in kwargs.items():
            setattr(self, key, value)

    def run(self, elements, copy_elements=True):
        """Run steps on each element in a given list.

        Args:
            elements (object or iterable of objects): Object or objects to
                operate on. Each element will be passed to each step in
                sequence.
            copy_elements (bool): Deep copy each element to ensure it does not
                mutate during run.

        Returns:
            output (list of objects): Objects returned from each step.
        """
        # Initialize output for new run.
        output = []

        # Make sure elements are contained in an iterable data structure.
        if not isinstance(elements, (tuple, list)):
            elements = (elements,)

        if not elements:
            self.logger.warning('No elements given.')
            return output

        for element in elements:
            # Create a deep copy of each element to pass to the steps insertion
            # point. The deep copy is to ensure the original data remains
            # unedited. The response value will be passed to each next step.
            # Using the same nomenclature to prevent renaming problems.
            if copy_elements:
                element = copy.deepcopy(element)

            # Initialize response before it's updated by the loop.
            response = None

            try:
                # Creates a generator with the given element that runs each
                # step in order and yields the result. The loop forces the
                # generator to continue working through all steps with the
                # given element. We don't need to handle any of the
                # intermediate responses, so we add a pass statement in the
                # loop. Because of scope leak and how Python is designed,
                # the variable response is updated with each loop and, after
                # the generator has worked through every step, response
                # contains the final response. That final response is then
                # saved for output.
                for response in self.walk(element):
                    pass
            except Exception as e:
                # Stop running workflow if it is not graceful.
                if not self.GRACEFUL:
                    self.logger.error(
                        f'Workflow {self.__class__.__name__} not graceful. '
                        'Halting for all elements.'
                    )
                    raise e
                else:
                    self.logger.info(
                        f'Workflow {self.__class__.__name__} is graceful. '
                        'Continuing for rest of elements.'
                    )
                    continue

            # Save last response from the generator if it's not empty.
            if response:
                output.append(response)

        # Return the output.
        return output

    def walk(self, element):
        """Walk through steps with a given element as a generator.

        The element is passed into a Step where it is interpreted and
        processed. The Step returns another object after processing which will
        then be passed into the next Step.

        This cycle of calling a Step with an element, catching the response,
        then calling the next Step with the response continues until there are
        no more Steps left to run.

        This method allows for a user to look at the intermediate output from
        each step to help debugging.

        Usage:
            walker = my_workflow.walk(blob)
            response = next(walker)

            Every time next(walker) is called, it runs the next step and yields
            the response for analysis.

        Args:
            element (object): Object operate on. It will be passed to each step
                in sequence.

        Yields:
            response (object): Object response from each step.
        """
        # The response value will be passed to each next step, so using the
        # same nomenclature.
        response = element

        for step in self._steps:
            if response:
                # Call the step, passing the previous response value to the
                # insertion point method while handling exceptions.
                try:
                    instance = step(workflow=self)
                    attempt = getattr(instance, instance.INSERTION_POINT)(
                        response
                    )
                    response = attempt

                    # Ensures method is a generator.
                    yield response
                except Exception as e:
                    instance.logger.exception(e)

                    # Stop running steps if step with error is not graceful,
                    # but keep running workflow on new elements.
                    if not step.GRACEFUL:
                        instance.logger.error(
                            f'Step {step.__name__} not graceful. Halting '
                            'steps for current element.'
                        )
                        raise e
                    else:
                        instance.logger.info(
                            f'Step {step.__name__} is graceful. Continuing '
                            'steps for current element.'
                        )
                        continue
            else:
                self.logger.warning(
                    f'Empty element. Skipping step {step.__name__}.'
                )
