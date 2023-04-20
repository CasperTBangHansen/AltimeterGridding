import time

class Timer:
    """Simple timer"""
    def __init__(self, print_string: str = "", start_on_init: bool = True):
        self.start_time = None
        self.end_time = None
        self.print_string = print_string
        if start_on_init:
            self.Start()
    
    def Start(self):
        self.start_time = time.perf_counter()

    def Stop(self):
        if self.start_time is None:
            raise ValueError("Start was not called before stop")

        self.end_time = time.perf_counter() - self.start_time
        print_string = f"({self.print_string})" if self.print_string else ""
        if self.end_time < 60:
            print(f"Time elapsed {print_string}: {self.end_time:.2f} s")
        if (self.end_time >= 60) & ((self.end_time)/60 < 60):
            print(f"Time elapsed {print_string}: {(self.end_time)/60:.2f} min")
        if ((self.end_time)/60 >= 60):
            print(f"Time elapsed {print_string}: {(self.end_time)/3600:.2f} h")