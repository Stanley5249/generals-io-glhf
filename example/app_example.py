import optimal_opening  # noqa: F401
from glhf.app import APP
from rich.console import Console

if __name__ == "__main__":
    console = Console()
    app = APP(console)
    app.server("socketio")
    app.add("OptimalOpening", default_room="test")
    app.add("OptimalOpening", default_room="test", userid="123", username="[Bot]123")
    app.gui(1)
    app.start()
