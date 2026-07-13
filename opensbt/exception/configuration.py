class RestrictiveConfigException(Exception):
    def __init__(self, message="Search configuration is too restrictive, no time for search available."):
        """"Initializes the exception.
        
        :param message: Message to be output.
        :type message:  str
        """
        self.message = message
        super().__init__(self.message)
