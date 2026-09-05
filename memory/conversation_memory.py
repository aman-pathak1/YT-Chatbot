from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Any


# ============================================================
# Conversation Message
# ============================================================

@dataclass(frozen=True)
class ConversationMessage:
    """
    Immutable representation of a single conversation message.
    """

    role: str
    content: str
    timestamp: str


# ============================================================
# Conversation Memory
# ============================================================

class ConversationMemory:
    """
    Production-ready session-level conversation memory.

    Responsibilities:
        - Store user/assistant messages
        - Keep memory bounded
        - Preserve complete conversation turns
        - Provide LLM-ready conversation history
        - Maintain session isolation
        - Support clear/reset
        - Provide serialization/export
        - Thread-safe access

    Important:
        This memory stores conversation context only.
        It is NOT the YouTube knowledge base.
    """

    VALID_ROLES = {"user", "assistant"}

    def __init__(
        self,
        session_id: str,
        max_turns: int = 10,
        max_history_characters: int = 12000,
    ) -> None:

        self._validate_session_id(session_id)

        if max_turns <= 0:
            raise ValueError(
                "max_turns must be greater than 0."
            )

        if max_history_characters <= 0:
            raise ValueError(
                "max_history_characters must be greater than 0."
            )

        self.session_id = session_id
        self.max_turns = max_turns
        self.max_history_characters = (
            max_history_characters
        )

        self._messages: list[ConversationMessage] = []

        # Protects memory when multiple requests access
        # the same session.
        self._lock = RLock()

    # ========================================================
    # Add User Message
    # ========================================================

    def add_user_message(
        self,
        content: str,
    ) -> None:

        self._add_message(
            role="user",
            content=content,
        )

    # ========================================================
    # Add Assistant Message
    # ========================================================

    def add_assistant_message(
        self,
        content: str,
    ) -> None:

        self._add_message(
            role="assistant",
            content=content,
        )

    # ========================================================
    # Add Complete Conversation Turn
    # ========================================================

    def add_turn(
        self,
        user_message: str,
        assistant_message: str,
    ) -> None:

        self._validate_content(user_message)
        self._validate_content(assistant_message)

        with self._lock:

            timestamp = self._timestamp()

            self._messages.append(
                ConversationMessage(
                    role="user",
                    content=user_message.strip(),
                    timestamp=timestamp,
                )
            )

            self._messages.append(
                ConversationMessage(
                    role="assistant",
                    content=assistant_message.strip(),
                    timestamp=self._timestamp(),
                )
            )

            self._trim()

    # ========================================================
    # Internal Add Message
    # ========================================================

    def _add_message(
        self,
        role: str,
        content: str,
    ) -> None:

        if role not in self.VALID_ROLES:
            raise ValueError(
                f"Invalid role: {role}"
            )

        self._validate_content(content)

        with self._lock:

            self._messages.append(
                ConversationMessage(
                    role=role,
                    content=content.strip(),
                    timestamp=self._timestamp(),
                )
            )

            self._trim()

    # ========================================================
    # Get All Messages
    # ========================================================

    def get_messages(
        self,
    ) -> list[ConversationMessage]:

        with self._lock:
            return list(self._messages)

    # ========================================================
    # Get Recent Messages
    # ========================================================

    def get_recent_messages(
        self,
        turns: int = 5,
    ) -> list[ConversationMessage]:

        if turns <= 0:
            return []

        with self._lock:

            message_count = turns * 2

            return list(
                self._messages[-message_count:]
            )

    # ========================================================
    # Get LLM-ready History
    # ========================================================

    def get_history(
        self,
        max_characters: int | None = None,
    ) -> str:

        with self._lock:

            if not self._messages:
                return ""

            character_limit = (
                max_characters
                if max_characters is not None
                else self.max_history_characters
            )

            if character_limit <= 0:
                return ""

            history_parts: list[str] = []
            total_characters = 0

            # Start from newest messages and work backwards.
            # This ensures recent context gets priority.
            for message in reversed(self._messages):

                role = self._format_role(
                    message.role
                )

                formatted = (
                    f"{role}: {message.content}"
                )

                message_length = len(formatted)

                if (
                    total_characters
                    + message_length
                    > character_limit
                ):
                    break

                history_parts.append(formatted)

                total_characters += (
                    message_length
                )

            history_parts.reverse()

            return "\n".join(history_parts)

    # ========================================================
    # Get History for Query Rewriter
    # ========================================================

    def get_rewriter_context(self) -> str:

        history = self.get_history()

        if not history:
            return ""

        return (
            "Previous conversation:\n"
            f"{history}"
        )

    # ========================================================
    # Check Empty
    # ========================================================

    def is_empty(self) -> bool:

        with self._lock:
            return len(self._messages) == 0

    # ========================================================
    # Number of Messages
    # ========================================================

    def message_count(self) -> int:

        with self._lock:
            return len(self._messages)

    # ========================================================
    # Number of Turns
    # ========================================================

    def turn_count(self) -> int:

        with self._lock:

            user_messages = sum(
                1
                for message in self._messages
                if message.role == "user"
            )

            return user_messages

    # ========================================================
    # Clear Conversation
    # ========================================================

    def clear(self) -> None:

        with self._lock:
            self._messages.clear()

    # ========================================================
    # Export Memory
    # ========================================================

    def to_dict(self) -> dict[str, Any]:

        with self._lock:

            return {
                "session_id": self.session_id,
                "max_turns": self.max_turns,
                "max_history_characters": (
                    self.max_history_characters
                ),
                "messages": [
                    asdict(message)
                    for message in self._messages
                ],
            }

    # ========================================================
    # Memory Statistics
    # ========================================================

    def get_stats(self) -> dict[str, int | str]:

        with self._lock:

            return {
                "session_id": self.session_id,
                "message_count": len(
                    self._messages
                ),
                "turn_count": self.turn_count(),
                "history_characters": len(
                    self.get_history()
                ),
            }

    # ========================================================
    # Trim Memory
    # ========================================================

    def _trim(self) -> None:

        # ----------------------------------------------------
        # First: keep complete user/assistant turns.
        # ----------------------------------------------------

        max_messages = self.max_turns * 2

        if len(self._messages) > max_messages:

            self._messages = self._messages[
                -max_messages:
            ]

        # ----------------------------------------------------
        # Second: enforce character budget.
        # ----------------------------------------------------

        while (
            len(self._messages) > 2
            and len(self.get_history())
            > self.max_history_characters
        ):

            # Remove the oldest complete turn.
            self._messages = self._messages[2:]

    # ========================================================
    # Validation
    # ========================================================

    @staticmethod
    def _validate_session_id(
        session_id: str,
    ) -> None:

        if (
            not session_id
            or not session_id.strip()
        ):
            raise ValueError(
                "session_id cannot be empty."
            )

    @staticmethod
    def _validate_content(
        content: str,
    ) -> None:

        if not content or not content.strip():
            raise ValueError(
                "Message content cannot be empty."
            )

    # ========================================================
    # Timestamp
    # ========================================================

    @staticmethod
    def _timestamp() -> str:

        return datetime.now(
            timezone.utc
        ).isoformat()

    # ========================================================
    # Role Formatting
    # ========================================================

    @staticmethod
    def _format_role(
        role: str,
    ) -> str:

        if role == "user":
            return "User"

        if role == "assistant":
            return "Assistant"

        return role.capitalize()

    # ========================================================
    # Representation
    # ========================================================

    def __len__(self) -> int:

        return self.message_count()

    def __repr__(self) -> str:

        return (
            "ConversationMemory("
            f"session_id='{self.session_id}', "
            f"messages={self.message_count()}, "
            f"turns={self.turn_count()}"
            ")"
        )