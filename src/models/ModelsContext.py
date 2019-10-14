from threading import Lock


class ModelsContext:
    pinsRetinanet = None
    pinsUnet = None
    rootSession = None
    rootGraph = None
    _lock = Lock()

    @classmethod
    def _create(cls):
        with cls._lock:
            pass

    accessor = _create

    def _read(self):
        pass

    def __enter__(self):
        # create model instances if not created
        pass

    def __exit__(self, *_):
        pass


if __name__ == '__main__':
    with ModelsContext():
        pass
