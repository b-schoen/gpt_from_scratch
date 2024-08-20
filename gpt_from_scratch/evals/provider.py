from typing import Protocol, ParamSpec

import evalugator
import evalugator.api
import evalugator.api.requests


P = ParamSpec("P")


# note: this could be a module or class
class Provider(Protocol):
    """Generic provider protocol, usually fufilled by a module in evalugator."""

    def provides_model(self, model_id: str) -> bool: ...

    def execute(
        self,
        model_id: str,
        request: evalugator.api.requests.Request,
    ) -> evalugator.api.requests.Response: ...

    def encode(self, model_id: str, *args: P.args, **kwargs: P.kwargs) -> str: ...

    def decode(self, model_id: str, *args: P.args, **kwargs: P.kwargs) -> str: ...
