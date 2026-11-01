"""Minimal stand-in for the parts of Strands that GoTrain imports.

Used only when strands-agents is not installed, so mocked tests can run on any
Python. Tests that need real SDK behavior skip unless the real package exists.
"""

import sys
import types


def real_strands_available():
    module = sys.modules.get("strands")
    if module is not None:
        return getattr(module, "__file__", None) is not None
    try:
        import strands  # noqa: F401
    except ImportError:
        return False
    return True


def install():
    if real_strands_available():
        return False
    if hasattr(sys.modules.get("strands.interrupt"), "InterruptException"):
        # Already installed: keep the same classes, since modules imported earlier
        # (agent.hitl) catch InterruptException by identity.
        return True
    module = sys.modules.get("strands") or types.ModuleType("strands")

    def tool(func=None, **_options):
        return func if func is not None else (lambda function: function)

    class ToolContext:
        pass

    class Interrupt:
        def __init__(self, id, name, reason=None, response=None):
            self.id, self.name, self.reason, self.response = id, name, reason, response

    class InterruptException(Exception):
        def __init__(self, interrupt):
            super().__init__(interrupt.name)
            self.interrupt = interrupt

    module.tool = tool
    module.Agent = getattr(module, "Agent", object)
    module.ToolContext = ToolContext
    models = sys.modules.get("strands.models") or types.ModuleType("strands.models")
    models.BedrockModel = getattr(models, "BedrockModel", object)
    module.models = models
    interrupt = types.ModuleType("strands.interrupt")
    interrupt.Interrupt = Interrupt
    interrupt.InterruptException = InterruptException
    sys.modules.update({"strands": module, "strands.models": models, "strands.interrupt": interrupt})
    return True
