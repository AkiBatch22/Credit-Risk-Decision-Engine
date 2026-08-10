"""Project-specific exceptions with useful source context."""

from __future__ import annotations

import traceback


class CreditRiskError(Exception):
    """A domain exception that retains the originating file and line."""

    def __init__(self, message: str, original_error: BaseException | None = None) -> None:
        self.original_error = original_error
        self.source_file: str | None = None
        self.source_line: int | None = None

        if original_error is not None and original_error.__traceback__ is not None:
            frame = traceback.extract_tb(original_error.__traceback__)[-1]
            self.source_file = frame.filename
            self.source_line = frame.lineno

        context = ""
        if self.source_file and self.source_line:
            context = f" ({self.source_file}:{self.source_line})"
        super().__init__(f"{message}{context}")

    @classmethod
    def from_exception(cls, message: str, error: BaseException) -> "CreditRiskError":
        if isinstance(error, cls):
            return error
        return cls(message, original_error=error)
