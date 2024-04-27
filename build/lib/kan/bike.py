class Bike:

    def __init__(self, name: str):
        self.name = name

    def ride(self, road : str) -> bool:
        print("Riding: %s on road: %s" % (self.name, road))
        return True