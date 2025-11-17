from .controller import Controller

class WebController(Controller):
    def __init__(self, get_control_state_func):
        super().__init__()
        self.get_control_state = get_control_state_func

    def update(self, model, data):
        state = self.get_control_state()

        self.stop()

        if state['forward'] > 0:
            self.forward()
        if state['backward'] > 0:
            self.backward()
        if state['left'] > 0:
            self.turn_left()
        if state['right'] > 0:
            self.turn_right()

        super().update(model, data)
