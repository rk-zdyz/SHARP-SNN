
import sys
import os


with open("diagnose_log.txt", "w") as f:
    f.write(f"Python Executable: {sys.executable}\n")
    f.write(f"Python Logic Version: {sys.version}\n")
    f.write("sys.path:\n")
    for p in sys.path:
        f.write(f"  {p}\n")

    try:
        import flask
        f.write(f"Flask imported: {flask.__file__}\n")
    except ImportError as e:
        f.write(f"Failed to import flask: {e}\n")

    try:
        import eventlet
        f.write(f"Eventlet imported: {eventlet.__file__}\n")
    except ImportError as e:
        f.write(f"Failed to import eventlet: {e}\n")
